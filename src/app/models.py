from sqlalchemy import Column, String, Float, Text
from .database import Base

class SvmResult(Base):
    __tablename__ = "results"

    id = Column(String, primary_key=True)
    method = Column(String, nullable=False)
    test_data = Column(Text, nullable=False)
    result = Column(Text, nullable=False)
    confidence = Column(Text, nullable=True)
    score = Column(Float, nullable=True)
    params = Column(String, nullable=True)