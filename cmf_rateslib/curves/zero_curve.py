from cmf_rateslib.curves.base_curve import BaseZeroCurve


class LinearInterpolator:
    def __init__(self):
        pass

    def fit(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def interpolate_point(self, y):
        index = len(list(filter(lambda x: x <= y, self.xs)))
        if len(list(filter(lambda x: x == y, self.xs))) == 0:
            point = self.ys[index - 1] + (
                    (self.ys[index] - self.ys[index - 1]) / (self.xs[index] - self.xs[index - 1])) * (
                            y - self.xs[index - 1])
        else:
            point = self.ys[index - 1]
        return point

    def interpolate_array(self, ys):
        inter_list = [self.interpolate_point(y) for y in ys]
        return inter_list


class QuadInterpolator:
    def __init__(self):
        pass

    def fit(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def quad_interpolate_point(self, y):
        index = len(list(filter(lambda x: x <= y, self.xs)))
        if index < len(self.xs) - 1:
            y_1 = self.ys[index - 1]
            y_2 = self.ys[index]
            y_3 = self.ys[index + 1]
            x_1 = self.xs[index - 1]
            x_2 = self.xs[index]
            x_3 = self.xs[index + 1]
            if len(list(filter(lambda x: x == y, self.xs))) == 0:
                a_2 = (y_3 - y_1) / ((x_3 - x_1) * (x_3 - x_2)) - (y_2 - y_1) / ((x_2 - x_1) * (x_3 - x_2))
                a_1 = (y_2 - y_1) / (x_2 - x_1) - a_2 * (x_2 + x_1)
                a_0 = y_1 - a_1 * x_1 - a_2 * x_1 ** 2
                point = a_0 + a_1 * y + a_2 * y ** 2
            else:
                point = self.ys[index - 1]
        else:
            y_1 = self.ys[index - 2]
            y_2 = self.ys[index - 1]
            y_3 = self.ys[index]
            x_1 = self.xs[index - 2]
            x_2 = self.xs[index - 1]
            x_3 = self.xs[index]
            if len(list(filter(lambda x: x == y, self.xs))) == 0:
                a_2 = (y_3 - y_1) / ((x_3 - x_1) * (x_3 - x_2)) - (y_2 - y_1) / ((x_2 - x_1) * (x_3 - x_2))
                a_1 = (y_2 - y_1) / (x_2 - x_1) - a_2 * (x_2 + x_1)
                a_0 = y_1 - a_1 * x_1 - a_2 * x_1 ** 2
                point = a_0 + a_1 * y + a_2 * y ** 2
            else:
                point = self.ys[index - 1]
        return point

    def quuad_interpolate_array(self, ys):
        inter_list = [self.quad_interpolate_point(y) for y in ys]
        return inter_list


class ZeroCurve(BaseZeroCurve):
    def __init__(self, maturities, rates):
        super().__init__(maturities, rates)

    def interpolation_type(self):
        print('linear_zero_curve', 'linear_log_df', 'linear_fwd_rates', 'quad_zero_curve', 'quad_log_df',
              'quad_fwd_rates')

    def interpolation(self, expiry, interp_type, tenor=None):
        inter = LinearInterpolator()
        inter_quad = QuadInterpolator()

        if interp_type == 'linear_zero_curve':
            inter.fit(self._maturities, self._rates)
            return inter.interpolate_point(expiry)

        elif interp_type == 'linear_log_df':
            inter.fit(self._maturities, -self._maturities * self._rates)
            return inter.interpolate_point(expiry)

        elif interp_type == 'linear_fwd_rates':
            return self.fwd_rate(expiry, tenor)

        elif interp_type == 'quad_zero_curve':
            inter_quad.fit(self._maturities, self._rates)
            return inter_quad.quad_interpolate_point(expiry)

        elif interp_type == 'quad_log_df':
            inter_quad.fit(self._maturities, -self._maturities * self._rates)
            return inter_quad.quad_interpolate_point(expiry)

        elif interp_type == 'quad_fwd_rates':
            inter_quad.fit(self._maturities, [self.fwd_rate(i, tenor) for i in self._maturities])
            return inter_quad.quad_interpolate_point(expiry)
        else:
            raise ValueError("unknown type of interpolation")

    pass
