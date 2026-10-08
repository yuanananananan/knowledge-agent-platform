import { useEffect, useState, useRef, useCallback } from 'react';
import { TrainingLogsStream } from '@/services/sse/training-logs-stream';
import { parseLogToMetric } from '@/services/sse/log-parser';
import type { Metric } from '@/types/task';

export function useTrainingLogsStream(jobId: string) {
  const [logs, setLogs] = useState<string[]>([]);
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [error, setError] = useState<Error | null>(null);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isCompleted, setIsCompleted] = useState(false);
  const streamRef = useRef<TrainingLogsStream | null>(null);

  // Initialize stream instance
  useEffect(() => {
    if (!streamRef.current) {
      streamRef.current = new TrainingLogsStream();
    }
  }, []);

  // Handle log appending
  const appendLog = useCallback((line: string) => {
    setIsConnecting(false); // 收到第一条日志说明已连接
    setLogs((prev) => [...prev, line]);

    // Parse log to metric if it contains epoch data
    const metric = parseLogToMetric(line);
    if (metric) {
      setMetrics((prev) => [...prev, metric]);
    }
  }, []);

  // Handle stream completion
  const handleComplete = useCallback(() => {
    setIsConnecting(false);
    setIsCompleted(true);
    setError(null);
  }, []);

  // Connect to SSE stream
  useEffect(() => {
    if (!jobId || !streamRef.current) {
      return;
    }

    setIsConnecting(true);
    setError(null);
    setLogs([]);
    setMetrics([]);
    setIsCompleted(false);

    streamRef.current.connect(jobId, {
      onLog: appendLog,
      onError: (err) => {
        setIsConnecting(false);
        setError(err);
      },
      onComplete: handleComplete,
    });

    // Cleanup on unmount or when jobId changes
    return () => {
      streamRef.current?.disconnect();
    };
  }, [jobId, appendLog, handleComplete]);

  // Manual reconnect function
  const reconnect = useCallback(() => {
    if (!jobId || !streamRef.current) return;

    setIsConnecting(true);
    setError(null);
    setLogs([]);
    setMetrics([]);
    setIsCompleted(false);

    streamRef.current.connect(jobId, {
      onLog: appendLog,
      onError: (err) => {
        setIsConnecting(false);
        setError(err);
      },
      onComplete: handleComplete,
    });
  }, [jobId, appendLog, handleComplete]);

  // Manual disconnect function
  const disconnect = useCallback(() => {
    streamRef.current?.disconnect();
  }, []);

  return {
    logs,
    metrics,
    error,
    isConnecting,
    isCompleted,
    reconnect,
    disconnect,
  };
}
