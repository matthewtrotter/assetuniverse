import logging
from datetime import datetime

from typing import List
from collections import defaultdict

import numpy as np
import pandas as pd
from ibapi.wrapper import EWrapper
from ibapi.client import EClient
from ibapi.common import TickerId, BarData
from ibapi.contract import Contract
from ibapi.utils import iswrapper

import datetime



ContractList = List[Contract]
BarDataList = List[BarData]



class TestWrapper(EWrapper):
    def __init__(self):
        super().__init__()


class TestClient(EClient):
    def __init__(self, wrapper):
        EClient.__init__(self, wrapper)


class TwsDownloadApp(TestWrapper, TestClient):
    def __init__(self, contracts: ContractList, start_date, end_date, max_days, duration, bar_size, data_type):
        TestClient.__init__(self, wrapper=self)
        TestWrapper.__init__(self)
        self.request_id = 0
        self.started = False
        self.next_valid_order_id = None
        self.contracts = contracts
        self.requests = {}
        self.bar_data = defaultdict(list)
        self.pending_ends = set()
        self.start_date = start_date
        self.end_date = end_date
        self.current = end_date
        self.duration = duration
        self.useRTH = 0
        self.barsize = bar_size
        self.data_type = data_type
        self.max_days = max_days

        dates = pd.date_range(self.start_date.strftime("%Y%m%d 00:00:00"), self.end_date.strftime("%Y%m%d 00:00:00"), freq='D')
        symbols = [self._get_symbol_of_contract(c) for c in contracts]
        self.closes = pd.DataFrame(data=None, index=dates, columns=symbols)

    def next_request_id(self, contract: Contract) -> int:
        self.request_id += 1
        self.requests[self.request_id] = contract
        return self.request_id

    def historicalDataRequest(self, contract: Contract) -> None:
        cid = self.next_request_id(contract)
        self.pending_ends.add(cid)

        self.reqHistoricalData(
            cid,  # tickerId, used to identify incoming data
            contract,
            self.current.strftime("%Y%m%d 00:00:00"),  # always go to midnight
            self.duration,  # amount of time to go back
            self.barsize,  # bar size
            self.data_type,  # historical data type
            self.useRTH,  # useRTH (regular trading hours)
            1,  # format the date in yyyyMMdd HH:mm:ss
            False,  # keep up to date after snapshot
            [],  # chart options
        )

    def daily_files(self):
        return SIZES.index(self.barsize.split()[1]) >= 5

    @iswrapper
    def headTimestamp(self, reqId: int, headTimestamp: str) -> None:
        contract = self.requests.get(reqId)
        ts = datetime.datetime.strptime(headTimestamp, "%Y%m%d  %H:%M:%S")
        startdate = self.start_date
        enddate = self.end_date
        # if isinstance(self.start_date, datetime.date):
        #     startdate = datetime.datetime.combine(self.start_date, datetime.datetime.min.time())
        #     enddate = datetime.datetime.combine(self.end_date, datetime.datetime.min.time())
        logging.info("Head Timestamp for %s is %s", contract, ts)
        if ts > startdate or self.max_days:
            logging.warning("Overriding start date, setting to %s", ts)
            self.start_date = ts  # TODO make this per contract
        if ts > enddate:
            logging.warning("Data for %s is not available before %s", contract, ts)
            self.done = True
            return
        # if we are getting daily data or longer, we'll grab the entire amount at once
        if self.daily_files():
            days = (enddate - startdate).days
            if days < 365:
                self.duration = "%d D" % days
            else:
                self.duration = "%d Y" % np.ceil(days / 365)
            # when getting daily data, look at regular trading hours only
            # to get accurate daily closing prices
            self.useRTH = 1
            # round up current time to midnight for even days
            # if isinstance(self.current, datetime.datetime):
            self.current = self.current.replace(
                hour=0, minute=0, second=0,# microsecond=0
            )

        self.historicalDataRequest(contract)

    @iswrapper
    def historicalData(self, reqId: int, bar) -> None:
        if pd.DatetimeIndex([str(bar.date)])[0] in self.closes.index:
            self.bar_data[reqId].append(bar)
            symbol = self._get_symbol_of_contract(self.requests[reqId])
            self.closes.loc[str(bar.date), symbol] = bar.close

    def _get_symbol_of_contract(self, contract):
        if contract.secType == 'FUT':
            return contract.localSymbol
        elif contract.secType == 'STK':
            return contract.symbol

    @iswrapper
    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:
        super().historicalDataEnd(reqId, start, end)
        self.pending_ends.remove(reqId)
        if len(self.pending_ends) == 0:
            self.current = datetime.datetime.strptime(start, "%Y%m%d  %H:%M:%S")
            if self.current <= self.start_date:
                self.done = True
            else:
                for contract in self.contracts:
                    self.historicalDataRequest(contract)

    @iswrapper
    def connectAck(self):
        logging.info("Connected")

    @iswrapper
    def nextValidId(self, order_id: int):
        super().nextValidId(order_id)

        self.next_valid_order_id = order_id
        logging.info(f"nextValidId: {order_id}")
        # we can start now
        self.start()

    def start(self):
        if self.started:
            return

        self.started = True
        for contract in self.contracts:
            self.reqHeadTimeStamp(
                self.next_request_id(contract), contract, self.data_type, 0, 1
            )

    @iswrapper
    def error(self, req_id: TickerId, error_code: int, error: str):
        super().error(req_id, error_code, error)
        if req_id < 0:
            logging.debug("Error. Id: %s Code %s Msg: %s", req_id, error_code, error)
        else:
            logging.error("Error. Id: %s Code %s Msg: %s", req_id, error_code, error)
            # we will always exit on error since data will need to be validated
            self.done = True


def make_contract(symbol: str, sec_type: str, currency: str, exchange: str, localsymbol: str = None) -> Contract:
    contract = Contract()
    contract.symbol = symbol
    contract.secType = sec_type
    contract.currency = currency
    contract.exchange = exchange
    if localsymbol:
        contract.localSymbol = localsymbol
    return contract


SIZES = ["secs", "min", "mins", "hour", "hours", "day", "week", "month"]
DURATIONS = ["S", "D", "W", "M", "Y"]


if __name__ == "__main__":
    symbols = ['S420U1']
    contracts = []

    contract = Contract()
    contract.secType = 'FUT'
    contract.currency = 'USD'
    contract.exchange = 'GLOBEX'
    contract.localSymbol = 'ESU1'
    contracts.append(contract)

    contract = Contract()
    contract.secType = 'FUT'
    contract.currency = 'USD'
    contract.exchange = 'ECBOT'
    contract.localSymbol = 'ZB   SEP 21'
    contracts.append(contract)

    contract = Contract()
    contract.symbol = 'SPY'
    contract.secType = 'STK'
    contract.currency = 'USD'
    contract.exchange = 'SMART'
    contracts.append(contract)


    now = datetime.datetime.now()
    start_date = now - datetime.timedelta(days=250)
    end_date = now

    app = TwsDownloadApp(contracts, start_date, end_date, False, '1 Y', '1 day', 'TRADES')
    app.connect("127.0.0.1", 7497, clientId=0)
    app.run()
    closes = app.closes
    print(closes)
