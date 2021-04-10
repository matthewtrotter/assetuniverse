#!/usr/bin/env python

"""
Title: Asset Universe Class Definition
Author: Matthew Trotter
Description:
    This asset universe class will download and parse stock and
    financial asset data from various online sources. You can use
    the price DataFrame or daily returns DataFrame for further
    analysis and testing.

Copyright 2019 Matthew Trotter
"""

from assetuniverse import AssetUniverse
import datetime
import numpy as np
from pandas import DataFrame

class Index:

    def __init__(self, start, end, symbols, weights, name):
        self.start = start
        self.end = end
        self.sym = symbols
        self.weights = weights/np.sum(weights)
        self.name = name

        # Assign weights and calculate normalized prices and returns
        self._calculatepricesandreturns()

    def calculateallocation(self, dollaramount):
        # Returns a Dataframe containing the dollar amount for each asset in the index based on the weights
        allocations = DataFrame(data=None, index=self.sym, columns=["Allocation ($)", "Allocation (Shares)"])
        allocations["Allocation ($)"] = dollaramount*self.weights
        allocations["Allocation (Shares)"] = allocations["Allocation ($)"]/self.original_prices.iloc[-1,:]
        return allocations


    def _calculatepricesandreturns(self):
        # Assign weights and calculate normalized prices and returns
        AU = AssetUniverse(self.start, self.end, self.sym)
        self.r = DataFrame((AU.r*self.weights).sum(axis=1), columns=[self.name,])
        #self.r.rename({"Closes": self.name})
        self.p = DataFrame(data=(1+self.r).cumprod(), columns=[self.name,])
        self.original_prices = AU.originalprices



if __name__ == "__main__":
    """days = 365.25 * 10
    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=days)
    #sym = ["VWELX", "DODBX", "Gold"] # Longest history
    sym = ["XOM", "AAPL"]
    weights = [100, 200]
    index0 = Index(start, end, sym, weights, "Test Index")
    print(index0.p)
    print(index0.r)"""

    days = 300
    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=days)
    # sym = ["VWELX", "DODBX", "Gold"] # Longest history
    sym = ["SBUX", "TLT", "Gold"]
    cashsym = "VFISX"
    borrow_spread = 2.0

    index1 = ["AAPL", "MSFT", "GOOG"]
    index1 = Index(start, end, index1, np.ones(len(index1)), "Index 1")
    index2 = ["XOM", "SLB", "RDS-B"]
    index2 = Index(start, end, index2, np.ones(len(index2)), "Index 2")
    indices = [index1, index2]

    AU = AssetUniverse(start, end, sym, indices=indices, offline=False, cashsym=cashsym, borrow_spread=borrow_spread)

    print(AU.p)
    print(index1.p)
    print(index2.p)





