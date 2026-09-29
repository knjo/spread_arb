import unittest

from ..common import calendar as cal


class CalendarTest(unittest.TestCase):
    def test_typhoon_known_only_after_publication(self):
        self.assertTrue(cal.is_planned_session("20260710", as_of="20260706"))
        self.assertFalse(cal.is_planned_session("20260710", as_of="20260713"))
        self.assertFalse(cal.is_planned_session("20260710", as_of="20260710"))

    def test_planned_holiday_and_weekend(self):
        self.assertFalse(cal.is_planned_session("20260619", as_of="20260504"))
        self.assertFalse(cal.is_planned_session("20260711", as_of="20260706"))  # Saturday

    def test_sessions_after(self):
        self.assertEqual(cal.sessions_after("20260706", "20260715", "20260706"),
                         ["20260707", "20260708", "20260709", "20260710", "20260713", "20260714", "20260715"])
        self.assertEqual(len(cal.sessions_after("20260706", "20260715", "20260713")), 6)
        self.assertEqual(cal.next_session("20260710", "20260706"), "20260713")
        self.assertEqual(cal.calendar_days("20260706", "20260715"), 9)


if __name__ == "__main__":
    unittest.main()
