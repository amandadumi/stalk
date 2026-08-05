from stalk.params.PesFunction import PesFunction
from stalk.params.ThermoResult import ThermoResult



class ThermoFunction(PesFunction):
    def __init__(self, func, args={}, quantity="enthalpy"):
        super().__init__(func, args, quantity=quantity)
        self.quantity = quantity


    def _evaluate_structure(
        self,
        structure,
        sigma=0.0,
        pressure=None,
        volume=None,
        var_eff_map=None,
        **kwargs
    ):
        eval_args = self.args.copy()
        eval_args.update(**kwargs)

        self._set_samples(structure, sigma=sigma, var_eff_map=var_eff_map)

        res = self.func(structure, sigma=sigma, **eval_args)

        if isinstance(res, ThermoResult):
            return res

        if isinstance(res, PesResult):
            return ThermoResult(
                energy=res.value,
                energy_error=res.error,
                pressure=pressure,
                volume=volume,
            )

        if isinstance(res, tuple) and len(res) == 2:
            value, error = res
            return ThermoResult(
                energy=value,
                energy_error=error,
                pressure=pressure,
                volume=volume,
            )

        raise TypeError(
            "ThermoFunction callable must return ThermoResult, PesResult, or (value, error)."
        )

    def evaluate(
        self,
        structure,
        sigma=0.0,
        add_sigma=False,
        quantity=None,
        pressure=None,
        volume=None,
        var_eff_map=None,
        **kwargs
    ):
        quantity = self.resolve_quantity(quantity, fallback="enthalpy")

        result = self._evaluate_structure(
            structure,
            sigma=sigma,
            pressure=pressure,
            volume=volume,
            var_eff_map=var_eff_map,
            **kwargs
        )
        self._load_structure(
            structure,
            result=result,
            sigma=sigma,
            add_sigma=add_sigma,
            quantity=quantity,
            var_eff_map=var_eff_map,
        )
        return result

    def _load_structure(
        self,
        structure,
        result=None,
        sigma=0.0,
        add_sigma=False,
        quantity=None,
        var_eff_map=None,
    ):
        if result is None:
            return
        
        quantity = self.resolve_quantity(quantity, fallback="enthalpy")

        if add_sigma:
            result.add_sigma(sigma)
        if hasattr(result, "use_quantity"):
            if quantity == "enthalpy":
                result.compute_enthalpy()
            result.use_quantity(quantity)

        structure.value = result.value
        structure.error = result.error
        self._update_var_eff_map(structure, var_eff_map=var_eff_map)
