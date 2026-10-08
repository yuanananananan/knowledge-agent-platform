import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { ScrollArea } from '../ui/scroll-area';
import { useValidationLogs, useValidationResult } from '@/hooks/val';

interface OfflineValResultProps {
  jobId: string;
}
export default function OfflineValResult({ jobId }: OfflineValResultProps) {
  const { data: valLog, isLoading: isLoadingLogs } = useValidationLogs(jobId);
  const { data: valResult, isLoading: isLoadingResult } = useValidationResult(jobId);
  const formatLog = (log: string) => {
    return log.split('\\n').map((line, index) => <div key={index}>{line}</div>);
  };

  return (
    <div className="space-y-2">
      <Card>
        <CardHeader>
          <CardTitle>测试结果</CardTitle>
          <CardDescription>测试已完成，以下是各测试指标</CardDescription>
        </CardHeader>
        <CardContent>
          {isLoadingResult ? (
            <div>Loading results...</div>
          ) : valResult?.metrics ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <div className="rounded-lg border bg-card p-4 shadow-sm">
                <div className="text-sm font-medium text-muted-foreground">Overall Accuracy</div>
                <div className="mt-2 text-2xl font-bold">{valResult.metrics.oa.toFixed(2)}%</div>
              </div>

              <div className="rounded-lg border bg-card p-4 shadow-sm">
                <div className="text-sm font-medium text-muted-foreground">F1 Score</div>
                <div className="mt-2 text-2xl font-bold">
                  {(valResult.metrics.f1 * 100).toFixed(2)}%
                </div>
              </div>

              <div className="rounded-lg border bg-card p-4 shadow-sm">
                <div className="text-sm font-medium text-muted-foreground">Kappa Coefficient</div>
                <div className="mt-2 text-2xl font-bold">
                  {(valResult.metrics.kappa * 100).toFixed(2)}%
                </div>
              </div>

              <div className="rounded-lg border bg-card p-4 shadow-sm">
                <div className="text-sm font-medium text-muted-foreground">Precision</div>
                <div className="mt-2 text-2xl font-bold">
                  {(valResult.metrics.precision * 100).toFixed(2)}%
                </div>
              </div>

              <div className="rounded-lg border bg-card p-4 shadow-sm">
                <div className="text-sm font-medium text-muted-foreground">Recall</div>
                <div className="mt-2 text-2xl font-bold">
                  {(valResult.metrics.recall * 100).toFixed(2)}%
                </div>
              </div>

              <div className="rounded-lg border bg-card p-4 shadow-sm">
                <div className="text-sm font-medium text-muted-foreground">Class Accuracy</div>
                <div className="mt-2 text-2xl font-bold">
                  {valResult.metrics.classAccuracy.join(' / ')}
                </div>
              </div>
            </div>
          ) : (
            <div>No results available</div>
          )}
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>测试日志</CardTitle>
          <CardDescription>测试已完成，日志如下</CardDescription>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-60 bg-accent rounded-md p-4">
            {isLoadingLogs ? (
              <div>Loading logs...</div>
            ) : valLog?.log ? (
              <pre className="whitespace-pre-wrap wrap-break-word font-mono text-sm">
                {formatLog(valLog.log)}
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
