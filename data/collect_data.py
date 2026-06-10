import asyncio
from bleak import BleakClient, BleakScanner
import csv

CHAR_UUID = "abcd1234-5678-1234-5678-1234567890ab"

label = int(input("Enter label (0=calm, 1=fidget): "))

file = open("data.csv", "a", newline="")
writer = csv.writer(file)

def handle_data(sender, data):
    try:
        line = data.decode()
        p, r = map(float, line.split(','))

        writer.writerow([p, r, label])
        print(p, r, label)

    except:
        pass

async def main():
    devices = await BleakScanner.discover()

    target = "C0:CD:D6:85:67:1E"
    if target is None:
        print("Device not found")
        return

    async with BleakClient(target) as client:
        print("Connected to ESP32")
        await client.start_notify(CHAR_UUID, handle_data)

        while True:
            await asyncio.sleep(1)

asyncio.run(main())