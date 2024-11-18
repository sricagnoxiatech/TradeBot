import asyncio
from internal.events import Events
from internal.pubsub import PubSub
from internal.strategy import StrategyMap
from internal.streams import Streams
from utils.env import Env
import time

async def main():
    max_retries = 5
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            # Format NATS URL properly
            url = f"nats://{Env.NATS_USER}:{Env.NATS_PASS}@{Env.NATS_URL}"
            
            instance = await PubSub.init(url)
            pubsub = PubSub(instance)
            
            # Set up strategy
            hashmap = StrategyMap()
            
            # Configure streams
            await pubsub.jetstream(Streams.DataFrame)

            async def handler(data):
                symbol = data['kline']['symbol']
                strategy = hashmap.get_instance(symbol)
                strategy.populate(data)
                payload = strategy.get_payload()
                await pubsub.publish(Events.DataFrame, payload)

            await pubsub.subscribe(Events.Kline, handler)
            
            # Keep the connection alive
            while True:
                await asyncio.sleep(1)
                
            return
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                raise
            time.sleep(retry_delay)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("Shutting down gracefully...")
    finally:
        loop.close()