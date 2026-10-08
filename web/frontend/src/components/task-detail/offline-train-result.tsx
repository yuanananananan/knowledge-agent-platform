import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { useTrainingLogs } from '@/hooks/train/use-training-logs';
import { useTrainingResult } from '@/hooks/train/use-training-result';
import { ScrollArea } from '../ui/scroll-area';
import { TrainingMetricsDisplay } from './training-metrics-display';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { TrainingMetricsChart } from './training-metrics-chart';
import { Separator } from '../ui/separator';

interface OfflineTrainResultProps {
  jobId: string;
}
export default function OfflineTrainResult({ jobId }: OfflineTrainResultProps) {
  const { data: trainLog, isLoading: isLoadingLogs } = useTrainingLogs(jobId);
  const { data: trainResult, isLoading: isLoadingResult } = useTrainingResult(jobId);
  const formatLog = (log: string) => {
    return log.split('\\n').map((line, index) => <div key={index}>{line}</div>);
  };

  return (
    <div className="space-y-2">
      <Card>
        <CardHeader>
          <CardTitle>训练结果</CardTitle>
          <CardDescription>训练已完成，以下是各训练指标</CardDescription>
        </CardHeader>
        <CardContent>
          {isLoadingResult ? (
            <div>Loading results...</div>
          ) : trainResult ? (
            <Tabs defaultValue="table-view">
              <TabsList>
                <TabsTrigger value="table-view">表格视图</TabsTrigger>
                <TabsTrigger value="chart-view">图表视图</TabsTrigger>
              </TabsList>
              <Separator className="my-3" />
              <TabsContent value="table-view">
                <TrainingMetricsDisplay trainResult={trainResult} />
              </TabsContent>
              <TabsContent value="chart-view">
                <TrainingMetricsChart metrics={trainResult.metrics} />
              </TabsContent>
            </Tabs>
          ) : (
            <div>No results available</div>
          )}
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>训练日志</CardTitle>
          <CardDescription>训练已完成，日志如下</CardDescription>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-60 bg-accent rounded-md p-4">
            {isLoadingLogs ? (
              <div>Loading logs...</div>
            ) : trainLog?.log ? (
              <pre className="whitespace-pre-wrap wrap-break-word font-mono text-sm">
                {formatLog(trainLog.log)}
              </pre>
            ) : (
              <div>No logs available</div>
            )}
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  );
}
