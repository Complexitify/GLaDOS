"""
    A utility for managing tp-link kasa smart home devices.
"""

import asyncio
import kasa

def get_plug(ip: str) -> kasa.SmartPlug:
    plug = None

    try:
        plug = kasa.SmartPlug(ip)
    except:
        print("Error getting plug at ip: " + ip)
    
    return plug