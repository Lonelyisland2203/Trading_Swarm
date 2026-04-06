"""
Production Binance execution client with comprehensive safety controls.

Wraps CCXT with:
- Testnet-first safety (live trading requires ALLOW_LIVE_TRADING=true env var)
- Order operations (limit, market, cancel, status)
- Position management (balance, open positions, close)
- Signal acceptance logic with 7-stage safety checks
- Comprehensive logging via StateManager
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Optional, List

import ccxt.async_support as ccxt
from loguru import logger

from config.fee_model import FeeModelSettings
from config.settings import ExecutionSettings
from execution.exceptions import (
    KillSwitchActiveError,
    DailyLossLimitError,
    PositionLimitError,
    CooldownActiveError,
    DailyTradeCountError,
    InsufficientBalanceError,
    LiveTradingNotAllowedError,
    OrderRejectedError,
)
from execution.models import (
    OrderResult,
    OrderStatus,
    Position,
    TradeDecision,
    SignalInput,
)
from execution.position_sizing import calculate_position_size
from execution.state_manager import StateManager


class BinanceExecutionClient:
    """
    Production-ready Binance execution client with comprehensive safety controls.

    Implements:
    - CCXT wrapper with async/await pattern
    - Kill switch for emergency halt
    - Daily loss circuit breaker
    - Position limits
    - Order cooldown
    - Confidence and return thresholds
    - Signal acceptance pipeline (7-stage safety checks)
    - State persistence via StateManager
    """

    def __init__(
        self,
        execution_settings: ExecutionSettings,
        fee_model_settings: FeeModelSettings,
        state_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize BinanceExecutionClient.

        Args:
            execution_settings: Execution configuration
            fee_model_settings: Fee model for position sizing
            state_dir: Optional override for state directory

        Raises:
            LiveTradingNotAllowedError: If live trading attempted without permission
        """
        self.execution_settings = execution_settings
        self.fee_model_settings = fee_model_settings

        # Check live trading authorization
        if not execution_settings.testnet:
            allow_live = os.getenv("ALLOW_LIVE_TRADING", "")
            if not allow_live or allow_live.lower() != "true":
                raise LiveTradingNotAllowedError()

        self.testnet = execution_settings.testnet

        # Initialize CCXT exchange
        self.exchange = getattr(ccxt, "binance")({
            "apiKey": execution_settings.api_key,
            "secret": execution_settings.api_secret,
            "enableRateLimit": True,
            "sandbox": execution_settings.testnet,
        })

        # Initialize state manager
        state_path = state_dir or execution_settings.state_dir
        self.state_manager = StateManager(state_path)

        logger.info(
            f"BinanceExecutionClient initialized (testnet={self.testnet}, "
            f"mode={execution_settings.mode})"
        )

    async def place_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
    ) -> OrderResult:
        """
        Place a limit order on the exchange.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            side: 'buy' or 'sell'
            amount: Order amount/quantity
            price: Limit price

        Returns:
            OrderResult with order details

        Raises:
            KillSwitchActiveError: If kill switch is active
            InsufficientBalanceError: If insufficient balance
            OrderRejectedError: If order rejected by exchange
        """
        # Safety check: kill switch
        if self.state_manager.is_kill_switch_active():
            raise KillSwitchActiveError()

        # Validate balance
        await self._check_balance(symbol, side, amount, price)

        try:
            # Place order
            if side == "buy":
                response = await self.exchange.create_limit_buy_order(
                    symbol, amount, price
                )
            else:
                response = await self.exchange.create_limit_sell_order(
                    symbol, amount, price
                )

            # Parse response
            order = self._parse_order_response(response, "limit")

            # Log order
            self.state_manager.log_order({
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side,
                "type": order.order_type,
                "amount": order.amount,
                "price": order.price,
                "status": order.status,
            })

            return order

        except Exception as e:
            logger.error(f"Failed to place limit order: {e}")
            raise OrderRejectedError(
                reason=f"Failed to place limit order: {str(e)}",
                exchange_error=e,
            )

    async def place_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
    ) -> OrderResult:
        """
        Place a market order on the exchange.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            side: 'buy' or 'sell'
            amount: Order amount/quantity

        Returns:
            OrderResult with order details

        Raises:
            KillSwitchActiveError: If kill switch is active
            InsufficientBalanceError: If insufficient balance
            OrderRejectedError: If order rejected by exchange
        """
        # Safety check: kill switch
        if self.state_manager.is_kill_switch_active():
            raise KillSwitchActiveError()

        # Get current price for balance check
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            current_price = ticker["last"]
        except Exception as e:
            logger.error(f"Failed to fetch ticker for {symbol}: {e}")
            raise OrderRejectedError(
                reason=f"Failed to fetch ticker: {str(e)}",
                exchange_error=e,
            )

        # Validate balance (use current price as estimate)
        await self._check_balance(symbol, side, amount, current_price)

        try:
            # Place order
            if side == "buy":
                response = await self.exchange.create_market_buy_order(symbol, amount)
            else:
                response = await self.exchange.create_market_sell_order(symbol, amount)

            # Parse response
            order = self._parse_order_response(response, "market")

            # Log order
            self.state_manager.log_order({
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side,
                "type": order.order_type,
                "amount": order.amount,
                "status": order.status,
            })

            return order

        except Exception as e:
            logger.error(f"Failed to place market order: {e}")
            raise OrderRejectedError(
                reason=f"Failed to place market order: {str(e)}",
                exchange_error=e,
            )

    async def cancel_order(
        self,
        order_id: str,
        symbol: str,
    ) -> OrderStatus:
        """
        Cancel an open order.

        Args:
            order_id: Order ID to cancel
            symbol: Trading pair

        Returns:
            OrderStatus with updated order info

        Raises:
            OrderRejectedError: If cancellation fails
        """
        if self.state_manager.is_kill_switch_active():
            raise KillSwitchActiveError()

        try:
            response = await self.exchange.cancel_order(order_id, symbol)
            return self._parse_order_status(response)
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise OrderRejectedError(
                reason=f"Failed to cancel order: {str(e)}",
                exchange_error=e,
            )

    async def get_order_status(
        self,
        order_id: str,
        symbol: str,
    ) -> OrderStatus:
        """
        Get current status of an order.

        Args:
            order_id: Order ID to check
            symbol: Trading pair

        Returns:
            OrderStatus with current order info
        """
        try:
            response = await self.exchange.fetch_order(order_id, symbol)
            return self._parse_order_status(response)
        except Exception as e:
            logger.error(f"Failed to fetch order status {order_id}: {e}")
            raise OrderRejectedError(
                reason=f"Failed to fetch order status: {str(e)}",
                exchange_error=e,
            )

    async def get_balance(self) -> dict:
        """
        Fetch account balance.

        Returns:
            Balance dict with asset details
        """
        try:
            return await self.exchange.fetch_balance()
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            raise OrderRejectedError(
                reason=f"Failed to fetch balance: {str(e)}",
                exchange_error=e,
            )

    async def get_open_positions(self) -> List[Position]:
        """
        Get all open positions.

        Returns:
            List of Position objects for open positions

        Raises:
            OrderRejectedError: If fetching positions fails
        """
        try:
            positions_data = await self.exchange.fetch_positions()
            positions = []

            for pos in positions_data:
                if pos.get("contracts", 0) > 0:
                    position = Position(
                        symbol=pos["symbol"],
                        side=pos["side"],
                        amount=pos.get("contracts", 0),
                        entry_price=pos.get("entryPrice", 0),
                        mark_price=pos.get("markPrice", 0),
                        unrealized_pnl=pos.get("unrealizedPnl", 0),
                        leverage=pos.get("leverage", 1),
                        liquidation_price=pos.get("liquidationPrice"),
                    )
                    positions.append(position)

            return positions

        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            raise OrderRejectedError(
                reason=f"Failed to fetch positions: {str(e)}",
                exchange_error=e,
            )

    async def close_position(self, symbol: str) -> OrderResult:
        """
        Close an open position with a market order.

        Args:
            symbol: Trading pair

        Returns:
            OrderResult for close order

        Raises:
            OrderRejectedError: If position not found or close fails
        """
        if self.state_manager.is_kill_switch_active():
            raise KillSwitchActiveError()

        # Get open positions
        positions = await self.get_open_positions()
        position_to_close = None

        for pos in positions:
            if pos.symbol == symbol:
                position_to_close = pos
                break

        if not position_to_close:
            raise OrderRejectedError(
                reason=f"No open position found for {symbol}"
            )

        # Close position with market order
        # If long, sell; if short, buy
        side = "sell" if position_to_close.side == "long" else "buy"
        return await self.place_market_order(symbol, side, position_to_close.amount)

    async def accept_signal(self, signal: SignalInput) -> TradeDecision:
        """
        Evaluate signal against safety checks and return trading decision.

        Implements 7-stage safety pipeline:
        1. Kill switch check
        2. Confidence threshold
        3. Expected return threshold
        4. Daily loss circuit breaker
        5. Position limit check
        6. Order cooldown
        7. Balance check

        Args:
            signal: SignalInput from signal generation pipeline

        Returns:
            TradeDecision with execute flag and reason
            Note: This method NEVER places orders - only returns decision
        """
        # Stage 1: Kill switch check
        if self.state_manager.is_kill_switch_active():
            return TradeDecision(
                execute=False,
                reason="Kill switch active - trading halted",
                symbol=signal.symbol,
            )

        # Stage 2: Confidence threshold
        if signal.confidence < self.execution_settings.min_confidence:
            return TradeDecision(
                execute=False,
                reason=f"Confidence {signal.confidence:.2%} below threshold "
                       f"{self.execution_settings.min_confidence:.2%}",
                symbol=signal.symbol,
            )

        # Stage 3: Expected return threshold
        if signal.expected_return_pct < self.execution_settings.min_expected_return_pct:
            return TradeDecision(
                execute=False,
                reason=f"Expected return {signal.expected_return_pct:.2f}% below threshold "
                       f"{self.execution_settings.min_expected_return_pct:.2f}%",
                symbol=signal.symbol,
            )

        # Get daily stats for remaining checks
        try:
            balance = await self.get_balance()
            total_balance = balance.get("total", {}).get("USDT", 0.0)
            daily_stats = self.state_manager.get_daily_stats(total_balance)
        except Exception as e:
            logger.error(f"Failed to get daily stats: {e}")
            return TradeDecision(
                execute=False,
                reason=f"Failed to fetch account state: {str(e)}",
                symbol=signal.symbol,
            )

        # Stage 4: Daily loss circuit breaker
        if daily_stats.daily_loss_pct >= self.execution_settings.max_daily_loss_pct:
            return TradeDecision(
                execute=False,
                reason=f"Daily loss {daily_stats.daily_loss_pct:.2f}% exceeds limit "
                       f"{self.execution_settings.max_daily_loss_pct:.2f}%",
                symbol=signal.symbol,
            )

        # Get open positions for position limit check
        try:
            open_positions = await self.get_open_positions()
        except Exception as e:
            logger.error(f"Failed to get open positions: {e}")
            return TradeDecision(
                execute=False,
                reason=f"Failed to fetch open positions: {str(e)}",
                symbol=signal.symbol,
            )

        # Stage 5: Position limit check
        if len(open_positions) >= self.execution_settings.max_open_positions:
            return TradeDecision(
                execute=False,
                reason=f"Position limit reached: {len(open_positions)}/{self.execution_settings.max_open_positions}",
                symbol=signal.symbol,
            )

        # Stage 6: Order cooldown
        if daily_stats.last_order_timestamp:
            time_since_last_order = datetime.now() - daily_stats.last_order_timestamp
            if time_since_last_order.total_seconds() < self.execution_settings.order_cooldown_seconds:
                seconds_remaining = (
                    self.execution_settings.order_cooldown_seconds
                    - time_since_last_order.total_seconds()
                )
                return TradeDecision(
                    execute=False,
                    reason=f"Order cooldown active: {seconds_remaining:.1f}s remaining",
                    symbol=signal.symbol,
                )

        # Daily trade count check
        if daily_stats.trade_count >= self.execution_settings.max_daily_trades:
            return TradeDecision(
                execute=False,
                reason=f"Daily trade limit reached: {daily_stats.trade_count}/{self.execution_settings.max_daily_trades}",
                symbol=signal.symbol,
            )

        # Stage 7: Balance check and position sizing
        try:
            # Get current price
            ticker = await self.exchange.fetch_ticker(signal.symbol)
            current_price = ticker.get("last", signal.entry_price or 50000.0)

            # Get free balance for quote currency
            usdt_balance = balance.get("USDT", {}).get("free", 0.0)

            # Determine side and stop loss price
            if signal.direction == "long":
                side = "buy"
                stop_price = current_price * (1 - signal.stop_loss_pct / 100)
            else:
                side = "sell"
                stop_price = current_price * (1 + signal.stop_loss_pct / 100)

            # Calculate position size
            position_size = calculate_position_size(
                balance=usdt_balance,
                risk_pct=self.execution_settings.max_position_pct * 0.5,  # Conservative
                entry_price=current_price,
                stop_price=stop_price,
                fee_model=self.fee_model_settings,
                max_position_pct=self.execution_settings.max_position_pct,
            )

            if position_size.amount <= 0:
                return TradeDecision(
                    execute=False,
                    reason=f"Calculated position size is zero: {position_size.reason}",
                    symbol=signal.symbol,
                )

            # Calculate prices
            entry_price = current_price
            take_profit_price = None
            if signal.take_profit_pct:
                if signal.direction == "long":
                    take_profit_price = current_price * (1 + signal.take_profit_pct / 100)
                else:
                    take_profit_price = current_price * (1 - signal.take_profit_pct / 100)

            # All checks passed - return execution decision
            return TradeDecision(
                execute=True,
                reason="Signal passed all safety checks",
                symbol=signal.symbol,
                side=side,
                amount=position_size.amount,
                price=entry_price,
                order_type="limit",
                stop_loss_price=stop_price,
                take_profit_price=take_profit_price,
            )

        except Exception as e:
            logger.error(f"Failed to calculate position size: {e}")
            return TradeDecision(
                execute=False,
                reason=f"Failed to calculate position: {str(e)}",
                symbol=signal.symbol,
            )

    # Private helper methods

    async def _check_balance(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
    ) -> None:
        """
        Check if account has sufficient balance for order.

        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            amount: Order amount
            price: Order price

        Raises:
            InsufficientBalanceError: If insufficient balance
        """
        balance = await self.get_balance()

        if side == "buy":
            # For buy orders, check quote currency (USDT)
            required = amount * price
            available = balance.get("USDT", {}).get("free", 0.0)
            asset = "USDT"
        else:
            # For sell orders, check base currency (from symbol)
            base_asset = symbol.split("/")[0]
            required = amount
            available = balance.get(base_asset, {}).get("free", 0.0)
            asset = base_asset

        if available < required:
            raise InsufficientBalanceError(
                required=required,
                available=available,
                asset=asset,
            )

    def _parse_order_response(self, response: dict, order_type: str) -> OrderResult:
        """
        Parse exchange order response into OrderResult.

        Args:
            response: Exchange API response dict
            order_type: 'limit' or 'market'

        Returns:
            OrderResult instance
        """
        timestamp_ms = response.get("timestamp", int(datetime.now().timestamp() * 1000))
        timestamp = datetime.fromtimestamp(timestamp_ms / 1000)

        return OrderResult(
            order_id=response["id"],
            symbol=response["symbol"],
            side=response["side"],
            order_type=order_type,
            amount=response.get("amount", 0.0),
            price=response.get("price"),
            status=response.get("status", "open"),
            filled=response.get("filled", 0.0),
            remaining=response.get("remaining", 0.0),
            cost=response.get("cost", 0.0),
            fee=response.get("fee", {}).get("cost", 0.0) if isinstance(response.get("fee"), dict) else response.get("fee", 0.0),
            timestamp=timestamp,
        )

    def _parse_order_status(self, response: dict) -> OrderStatus:
        """
        Parse exchange order response into OrderStatus.

        Args:
            response: Exchange API response dict

        Returns:
            OrderStatus instance
        """
        return OrderStatus(
            order_id=response["id"],
            symbol=response["symbol"],
            status=response.get("status", "unknown"),
            filled=response.get("filled", 0.0),
            remaining=response.get("remaining", 0.0),
            average_price=response.get("average"),
        )
