import unittest

from ...ev_lookup_cost.causal_lookup import open_ns
from .messages import counts


class MessageTests(unittest.TestCase):
    def test_explicit_and_effective_cancel_are_one_send_and_lost_race_request_remains(self):
        day='20260724'
        ns=open_ns(day)+1_000_000_000
        trace=[dict(kind='event_cancel',position_id='a',ns=ns,stream='S2'),
               dict(kind='cancel',position_id='a',purpose='entry',ns=ns+50_000_000,stream='S2'),
               dict(kind='capacity_cancel',position_id='b',ns=ns,stream='S2')]
        result=counts(day,trace)
        self.assertEqual(result['F_outbound_messages'],2)

    def test_s2_position_stock_exit_and_future_hedge_use_different_venues(self):
        day='20260724'
        ns=open_ns(day)+1_000_000_000
        trace=[dict(kind='exit_quote',position_id='a',ns=ns,stream='S2'),
               dict(kind='taker_fill',position_id='a',purpose='exit_future',ns=ns+50_000_000,stream='S2')]
        result=counts(day,trace)
        self.assertEqual(result['S_outbound_messages'],1)
        self.assertEqual(result['F_outbound_messages'],1)


if __name__=='__main__':
    unittest.main()
