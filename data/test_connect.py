import asyncio
from bleak import BleakScanner, BleakClient

TARGET_NAME = "AnxietyDevice"   # from your ESP32 code

async def main():
    print("Scanning...")
    devices = await BleakScanner.discover()

    target = None
    for d in devices:
        print(d.name, d.address)
        if d.name == TARGET_NAME:
            target = d

    if target is None:
        print("Device not found!")
        return

    print("Connecting to:", target.address)

    async with BleakClient(target) as client:
        print("✅ Connected successfully!")

asyncio.run(main())