"""Tests for Oracleai."""
from src.core import Oracleai
def test_init(): assert Oracleai().get_stats()["ops"] == 0
def test_op(): c = Oracleai(); c.detect(x=1); assert c.get_stats()["ops"] == 1
def test_multi(): c = Oracleai(); [c.detect() for _ in range(5)]; assert c.get_stats()["ops"] == 5
def test_reset(): c = Oracleai(); c.detect(); c.reset(); assert c.get_stats()["ops"] == 0
def test_service_name(): c = Oracleai(); r = c.detect(); assert r["service"] == "oracleai"
