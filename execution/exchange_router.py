"""
Exchange router for dispatching to different exchange adapters.

Routes to HyperliquidAdapter or BinanceExecutionClient based on EXCHANGE env var.
Provides identical interface regardless of underlying exchange.

Usage:
    router = ExchangeRouter(...)
    await router.place_order(...)  # Routes to correct adapter

Config:
    EXCHANGE env var: "hyperliquid" | "binance" (default: "binance")
"""

import os
from typing import List, Optional

from loguru import logger

from execution.models import OrderResult, OrderStatus, Position


class ExchangeRouter:
    """
    Routes execution calls to the appropriate exchange adapter.

    Provides identical interface regardless of exchange:
    - place_order
    - cancel_order
    - get_positions
    - get_balance
    - flatten_all

    Exchange selection via EXCHANGE env var.
    """

    SUPPORTED_EXCHANGES = {"binance", "hyperliquid"}

    def __init__(
        self,
        private_key: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = True,
        state_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize ExchangeRouter.

        Args:
            private_key: Private key for Hyperliquid (EIP-712)
            api_key: API key for Binance
            api_secret: API secret for Binance
            testnet: Use testnet (default True)
            state_dir: Directory for state files

        Raises:
            ValueError: If EXCHANGE env var is unsupported
        """
        self._private_key = private_key
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet
        self._state_dir = state_dir

        # Get exchange selection from env
        exchange = os.environ.get("EXCHANGE", "binance").lower()

        if exchange not in self.SUPPORTED_EXCHANGES:
            raise ValueError(
                f"Unsupported exchange: {exchange}. "
                f"Supported: {', '.join(self.SUPPORTED_EXCHANGES)}"
            )

        self._current_exchange = exchange
        self._adapter = self._create_adapter(exchange)

        logger.info(f"ExchangeRouter initialized with exchange={exchange}")

    @property
    def current_exchange(self) -> str:
        """Return currently selected exchange."""
        return self._current_exchange

    def _create_adapter(self, exchange: str):
        """Create adapter for specified exchange."""
        if exchange == "hyperliquid":
            from execution.hyperliquid_adapter import HyperliquidAdapter

            return HyperliquidAdapter(
                private_key=self._private_key,
                testnet=self._testnet,
                state_dir=self._state_dir,
            )
        elif exchange == "binance":
            from execution.binance_client import BinanceExecutionClient
            from config.fee_model import FeeModelSettings
            from config.settings import ExecutionSettings

            # Create settings for Binance client
            exec_settings = ExecutionSettings(
                api_key=self._api_key or "",
                api_secret=self._api_secret or "",
                testnet=self._testnet,
            )
            fee_settings = FeeModelSettings()

            return BinanceExecutionClient(
                execution_settings=exec_settings,
                fee_model_settings=fee_settings,
                state_dir=self._state_dir,
            )
        else:
            raise ValueError(f"Unknown exchange: {exchange}")

    def switch_exchange(
        self,
        exchange: str,
        private_key: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ) -> None:
        """
        Switch to a different exchange at runtime.

        Args:
            exchange: Target exchange ("hyperliquid" or "binance")
            private_key: Private key for Hyperliquid
            api_key: API key for Binance
            api_secret: API secret for Binance
        """
        if exchange not in self.SUPPORTED_EXCHANGES:
            raise ValueError(f"Unsupported exchange: {exchange}")

        # Update credentials if provided
        if private_key:
            self._private_key = private_key
        if api_key:
            self._api_key = api_key
        if api_secret:
            self._api_secret = api_secret

        self._current_exchange = exchange
        self._adapter = self._create_adapter(exchange)

        logger.info(f"Switched to exchange={exchange}")

    async def place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        order_type: str = "limit",
        stop_loss_price: Optional[float] = None,
    ) -> OrderResult:
        """
        Place order via current exchange adapter.

        Args:
            symbol: Trading pair/coin
            side: "buy" or "sell"
            amount: Order amount
            price: Limit price (required for limit orders)
            order_type: "limit" or "market"
            stop_loss_price: Optional stop-loss price

        Returns:
            OrderResult with order details
        """
        if self._current_exchange == "hyperliquid":
            return await self._adapter.place_order(
                symbol=symbol,
                side=side,
                amount=amount,
                price=price,
                order_type=order_type,
                stop_loss_price=stop_loss_price,
            )
        else:
            # Binance adapter has different method names
            if order_type == "market":
                return await self._adapter.place_market_order(
                    symbol=symbol,
                    side=side,
                    amount=amount,
                )
            else:
                return await self._adapter.place_limit_order(
                    symbol=symbol,
                    side=side,
                    amount=amount,
                    price=price,
                )

    async def cancel_order(
        self,
        order_id: str,
        symbol: str,
    ) -> OrderStatus:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel
            symbol: Trading pair/coin

        Returns:
            OrderStatus with updated info
        """
        return await self._adapter.cancel_order(order_id, symbol)

    async def get_positions(self) -> List[Position]:
        """
        Get all open positions.

        Returns:
            List of Position objects
        """
        if self._current_exchange == "hyperliquid":
            return await self._adapter.get_positions()
        else:
            return await self._adapter.get_open_positions()

    async def get_balance(self) -> dict:
        """
        Get account balance.

        Returns:
            Balance dict (format may vary by exchange)
        """
        return await self._adapter.get_balance()

    async def flatten_all(self) -> None:
        """
        Close all open positions.

        Used for kill switch and emergency exit.
        """
        if self._current_exchange == "hyperliquid":
            await self._adapter.flatten_all()
        else:
            # Binance: close each position manually
            positions = await self.get_positions()
            for pos in positions:
                try:
                    await self._adapter.close_position(pos.symbol)
                except Exception as e:
                    logger.error(f"Failed to close {pos.symbol}: {e}")
