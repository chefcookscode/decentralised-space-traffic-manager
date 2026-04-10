"""
Inter-Satellite Communication Protocol (ISCP)
=============================================
A standardized, lightweight protocol for satellites from different operators
to exchange critical state and intent data.
"""

from iscp.payload import (
    ISCPPayload,
    ManeuverIntent,
    ManeuverType,
    PropulsionType,
    PAYLOAD_SIZE,
    MAX_PAYLOAD_BYTES,
)
