import numpy
import pandas


class LGBMImputer(object):

    def __init__(self, gbm):
        self.gbm = gbm

    def fit(self, x_data, t_data, **kwargs):
        x_data = numpy.array(x_data)
        t_data = numpy.array(t_data)

        self.gbm.fit(x_data, t_data, **kwargs)

        if 'eval_set' in kwargs and kwargs['eval_set']:
            x, t = kwargs['eval_set'][0]
            x, t = numpy.array(x), numpy.array(t)
        else:
            x, t = x_data, t_data

        # the -1 th column is for offset
        contrib_scores = self.gbm.predict(x, pred_contrib=True)[:,:-1]

        num_null = pandas.isnull(x).sum(axis=0)

        fill_values = []
        for i, nn in enumerate(num_null):
            if nn == 0:
                fill_values.append(None)
            elif nn == x.shape[0]:
                fill_values.append(0)
            else:
                fill_value = self.compute_fill_value(x[:,i], contrib_scores[:,i])
                fill_values.append(fill_value)

        self.fill_values_ = fill_values

    def transform(self, x_data, **kwargs):
        x_data = numpy.array(x_data)
        for i, v in enumerate(self.fill_values_):
            if v is None:
                continue
            x_data[numpy.isnan(x_data[:,i]),i] = v
        return x_data

    def compute_fill_value(self, x_values, t_values):
        df = pandas.DataFrame({'x_values': x_values, 't_values': t_values})

        null_mean = df[df['x_values'].isnull()]['t_values'].mean()

        df_nn = df[~df['x_values'].isnull()].copy()
        df_nn['t_sim'] = numpy.exp(-(df_nn['t_values'] - null_mean) ** 2)

        x_mean = (df_nn['x_values'] * df_nn['t_sim']).sum() / df_nn['t_sim'].sum()

        return x_mean
