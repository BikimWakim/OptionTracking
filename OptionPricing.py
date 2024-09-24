import numpy as np
from scipy.stats import norm
import pandas as pd
from datetime import datetime
import os
from config import FILE_NAME

#Black-Scholes Way
"""
C = price of the call option
P = price of the put option
S = current price of the underlying asset
X = strike price of the option
r = risk-free interest rate
q = dividend yield = 0 for Index
T = time to maturity(in years)
N() = norm.cdf() = cumulative distribution function of the standard normal distribution
sigma = volatility of the underlying asset
d1 = (ln(S_0/X)+(r+sigma^2/2)*T)/(sigma*sqrt(T))
d2 = d1 - sigma*sqrt(T)
"""

class BlackScholesOptions:
    def __init__(self, S, X, r, T, sigma):
        self.S = S
        self.X = X
        self.r = r
        self.T = T
        self.sigma = sigma

    def _calculate_d1_d2(self):
        d1 = (np.log(self.S/self.X)+(self.r + self.sigma**2 * 0.5)* self.T)/(self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    def black_scholes_call(self):
        d1, d2 = self._calculate_d1_d2()
        call_price = self.S  * norm.cdf(d1) - self.X * np.exp(-self.r * self.T) * norm.cdf(d2)
        return np.round(call_price,2)


    def black_scholes_put(self):
        d1, d2 = self._calculate_d1_d2()
        put_price = self.X * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        return np.round(put_price, 2)

    #Greeks
    def delta_call(self):
        d1, _ = self._calculate_d1_d2()
        delta_c = norm.cdf(d1)
        return np.round(delta_c, 2)

    def delta_put(self):
        d1, _ = self._calculate_d1_d2()
        delta_p = -norm.cdf(-d1)
        return np.round(delta_p, 2)

    def gamma(self):
        d1, _ = self._calculate_d1_d2()
        gamma_option = 1/(self.S * self.sigma * np.sqrt(self.T)) * norm.pdf(d1)
        return np.round(gamma_option, 2)

    def theta_call(self):
        d1, d2 = self._calculate_d1_d2()
        theta_c = (-((self.S * self.sigma*norm.pdf(d1))/(2*np.sqrt(self.T)))) - self.r * self.X * np.exp(-self.r * self.T) * norm.cdf(d2) 
        return np.round(theta_c, 2)

    def theta_put(self):
        d1, d2 = self._calculate_d1_d2()
        theta_p = (-((self.S * self.sigma * norm.pdf(d1))/(2 * np.sqrt(self.T)))) + self.r * self.X * np.exp(-self.r * self.T) * norm.cdf(-d2) 
        return np.round(theta_p, 2)

    def vega(self):
        d1, _ = self._calculate_d1_d2()
        vega_option = self.S * np.sqrt(self.T) * norm.pdf(d1)
        return np.round(vega_option, 2)

    def rho_call(self):
        d1, d2 = self._calculate_d1_d2()
        rho_c = self.X * self.T * np.exp(-self.r*self.T) * norm.cdf(d2)
        return np.round(rho_c, 2)
    
    def rho_put(self):
        d1, d2 = self._calculate_d1_d2()
        rho_p = -self.X * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)
        return np.round(rho_p, 2)

    def vanna(self):
        d1, _ = self._calculate_d1_d2()
        vanna = self.S * d1 * self.T / self.sigma
        return np.round(vanna, 2)

    def volga(self):
        d1, d2 = self._calculate_d1_d2()
        volga = self.S * np.sqrt(self.T) * d1 * d2
        return np.round(volga, 2)
    
    def parameters(self):
        return {'S': self.S, 'r': self.r, 'T': self.T, 'sigma': self.sigma}


columns = ['Date', 'Call Price', 'Put Price', 'Delta Call', 'Delta Put', 'Gamma', 'Theta Call', 'Theta Put', 'Vega', 'Rho Call', 'Rho Put', 'Vanna', 'Volga', 'Parameters']
file_name = FILE_NAME


def load_existing_df(file_name):
    if os.path.exists(file_name):
        try:
            df = pd.read_csv(file_name)
            print("Df loaded successfully")
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            df = pd.DataFrame(columns=columns)
    else:
        print("Df not found")
        df = pd.DataFrame(columns=columns)

    return df


def update_dataframe(S, X, r, T, sigma):
    df = load_existing_df(file_name)
    model = BlackScholesOptions(S, X, r, T, sigma)

    today = datetime.now().strftime('%Y-%m-%d')

    if today not in df['Date'].values:

        call_price = model.black_scholes_call()
        put_price = model.black_scholes_put()
        delta_call = model.delta_call()
        delta_put = model.delta_put()
        gamma = model.gamma()
        theta_call = model.theta_call()
        theta_put = model.theta_put()
        rho_call = model.rho_call()
        rho_put = model.rho_put()
        vega = model.vega()
        vanna = model.vanna()
        volga = model.volga()
        parameters = model.parameters()

        new_row = {
            'Date': today,
            'Call Price': call_price,
            'Put Price': put_price,
            'Delta Call': delta_call,
            'Delta Put': delta_put,
            'Gamma': gamma,
            'Theta Call': theta_call,
            'Theta Put': theta_put,
            'Rho Call': rho_call,
            'Rho Put': rho_put,
            'Vega': vega,
            'Vanna': vanna,
            'Volga': volga,
            'Parameters': parameters
        }

        new_df = pd.DataFrame([new_row])
        updated_df = pd.concat([df, new_df], ignore_index=True)
    
    else:
        print(f"Data for {today} already exists. Skipping update.")

    return updated_df

#Save DF to csv

def save_to_csv(df, file_name=FILE_NAME):
    df.to_csv(file_name, index=False)

df_update = update_dataframe(S = 5718.26, X = 5475, r = 0.0491, T = 0.06575, sigma = 0.173)
save_to_csv(df_update)


