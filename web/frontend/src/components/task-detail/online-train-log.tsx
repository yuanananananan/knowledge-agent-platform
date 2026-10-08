import { useEffect, useRef } from 'react';
import { useTrainingLogsStream } from '@/hooks/train/use-training-logs-stream';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { TrainingMetricsChart } from './training-metrics-chart';
import { ScrollArea } from '../ui/scroll-area';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { RefreshCw, Wifi, WifiOff, Loader2, CheckCircle } from 'lucide-react';

interface OnlineTrainLogProps {
  jobId: string;
}

export default function OnlineTrainLog({ jobId }: OnlineTrainLogProps) {
  const { logs, metrics, error, isConnecting, isCompleted, reconnect } =
    useTrainingLogsStream(jobId);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  // Auto-scroll log area to bottom when new logs arrive
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current.querySelector(
        '[data-radix-scroll-area-viewport]'
      );
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      }
    }
  }, [logs]);

  // Get connection status info
  const getConnectionStatus = () => {
    if (isConnecting) {
      return {
        icon: <Loader2 className="h-3 w-3 animate-spin" />,
        text: '连接中',
        variant: 'secondary' as const,
      };
    }
    if (error) {
      return {
        icon: <WifiOff className="h-3 w-3" />,
        text: '连接失败',
        variant: 'destructive' as const,
      };
    }
    if (isCompleted) {
      return {
        icon: <CheckCircle className="h-3 w-3" />,
        text: '已完成',
        variant: 'outline' as const,
      };
    }
    return {
      icon: <Wifi className="h-3 w-3" />,
      text: '已连接',
      variant: 'default' as const,
    };
  };

  const connectionStatus = getConnectionStatus();

  // Get latest metrics summary
  const latestMetric = metrics[metrics.length - 1];

  return (
    <div className="space-y-4">
      {/* Metrics Summary Card */}
      {latestMetric && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>训练进度</CardTitle>
                <CardDescription>当前训练轮次的最新指标</CardDescription>
              </div>
              <Badge variant="outline" className="text-sm">
                Epoch {latestMetric.epoch}
              </Badge>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="space-y-1">
                <div className="text-xs text-muted-foreground">OA</div>
                <div className="text-2xl font-bold">{(latestMetric.oa * 100).toFixed(2)}%</div>
              </div>
              <div className="space-y-1">
                <div className="text-xs text-muted-foreground">Kappa</div>
                <div className="text-2xl font-bold">{latestMetric.kappa.toFixed(4)}</div>
              </div>
              <div className="space-y-1">
                <div className="text-xs text-muted-foreground">F1 Score</div>
                <div className="text-2xl font-bold">{(latestMetric.f1 * 100).toFixed(2)}%</div>
              </div>
              <div className="space-y-1">
                <div className="text-xs text-muted-foreground">Loss</div>
                <div className="text-2xl font-bold">{latestMetric.loss.toFixed(4)}</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Metrics Chart Card */}
      <Card>
        <CardHeader>
          <CardTitle>训练指标</CardTitle>
          <CardDescription>展示训练过程中的实时指标变化趋势</CardDescription>
        </CardHeader>
        <CardContent>
          {metrics.length > 0 ? (
            <TrainingMetricsChart metrics={metrics} />
          ) : (
            <div className="h-[400px] flex items-center justify-center text-muted-foreground">
              等待训练数据...
            </div>
          )}
        </CardContent>
      </Card>

      {/* Training Logs Card */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <CardTitle>训练日志</CardTitle>
              <CardDescription>展示训练过程中的实时日志输出</CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant={connectionStatus.variant} className="gap-1">
                {connectionStatus.icon}
                {connectionStatus.text}
              </Badge>
              {error && (
                <Button size="sm" variant="outline" onClick={reconnect}>
                  <RefreshCw className="h-4 w-4 mr-1" />
                  重连
                </Button>
              )}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {error ? (
            <div className="rounded-md bg-destructive/10 border border-destructive/20 p-4">
              <div className="flex items-start gap-2">
                <WifiOff className="h-5 w-5 text-destructive mt-0.5" />
                <div className="flex-1">
                  <div className="font-medium text-destructive">连接失败</div>
                  <div className="text-sm text-muted-foreground mt-1">{error.message}</div>
                  <Button size="sm" variant="outline" onClick={reconnect} className="mt-2">
                    <RefreshCw className="h-4 w-4 mr-1" />
                    重新连接
                  </Button>
                </div>
              </div>
            </div>
          ) : (
            <ScrollArea ref={scrollAreaRef} className="h-96 bg-accent rounded-md p-4">
              {logs.length === 0 ? (
                <div className="flex items-center justify-center h-full text-muted-foreground">
                  {isConnecting ? '正在连接...' : '等待日志输出...'}
                </div>
              ) : (
                <div className="font-mono text-sm space-y-0.5">
                  {logs.map((log, index) => (
                    <div
                      key={index}
                      className="whitespace-pre-wrap wrap-break-word py-0.5 text-slate-50 px-2 -mx-2 rounded"
                    >
                      {log}
                    </div>
                  ))}
                </div>
              )}
            </ScrollArea>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
