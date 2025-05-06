from pylsl import StreamInlet, resolve_stream

def main():
    print("Looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')
    print(f"Found {len(streams)} stream(s)! Connecting to the first one.")
    inlet = StreamInlet(streams[0])
    print("Now pulling samples...\nPress Ctrl+C to stop.")

    try:
        while True:
            sample, timestamp = inlet.pull_sample()
            print(f"Timestamp: {timestamp:.5f}, Sample: {sample}")
    except KeyboardInterrupt:
        print("\nProgram interrupted. Exiting.")

if __name__ == '__main__':
    main()