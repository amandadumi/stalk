class QuantityMixin:
    _quantity = None

    @property
    def quantity(self):
        return self._quantity

    @quantity.setter
    def quantity(self, quantity):
        if quantity is None:
            self._quantity = None
        elif quantity in ("energy", "enthalpy"):
            self._quantity = quantity
        else:
            raise ValueError(
                f"quantity must be None, 'energy', or 'enthalpy', got {quantity}"
            )

    def resolve_quantity(self, quantity=None, pes=None):
        if quantity is not None:
            return quantity
        if self.quantity is not None:
            return self.quantity
        if pes is not None and hasattr(pes, "quantity") and pes.quantity is not None:
            return pes.quantity
        return "energy"