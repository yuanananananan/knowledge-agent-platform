/**
 * SSE (Server-Sent Events) client for streaming training logs
 */

export type SSECallbacks = {
  onLog?: (line: string) => void;
  onError?: (error: Error) => void;
  onComplete?: () => void;
};

export class TrainingLogsStream {
  private eventSource: EventSource | null = null;
  private baseUrl: string;
  private hasReceivedData = false;

  constructor() {
    this.baseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:3000';
  }

  /**
   * Connect to SSE endpoint and start streaming logs
   */
  connect(jobId: string, callbacks: SSECallbacks): void {
    // Close existing connection if any
    this.disconnect();

    this.hasReceivedData = false;
    const url = `${this.baseUrl}/api/train/logs/${jobId}`;
    this.eventSource = new EventSource(url);

    // Handle incoming messages
    this.eventSource.onmessage = (event) => {
      this.hasReceivedData = true;
      callbacks.onLog?.(event.data);
    };

    // Handle errors
    this.eventSource.onerror = () => {
      // If we've received data, treat connection close as normal completion
      // EventSource fires onerror when the server closes the connection
      if (this.hasReceivedData) {
        callbacks.onComplete?.();
      } else {
        // Only treat it as an error if we haven't received any data
        callbacks.onError?.(new Error('SSE connection error'));
      }
      this.disconnect();
    };
  }

  /**
   * Disconnect from SSE endpoint
   */
  disconnect(): void {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
  }

  /**
   * Check if currently connected
   */
  isConnected(): boolean {
    return this.eventSource !== null && this.eventSource.readyState === EventSource.OPEN;
  }
}
