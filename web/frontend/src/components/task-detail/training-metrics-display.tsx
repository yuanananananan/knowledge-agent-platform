import type { TrainingResult } from '@/types/task';

interface TrainingMetricsDisplayProps {
  trainResult: TrainingResult;
}

export function TrainingMetricsDisplay({ trainResult }: TrainingMetricsDisplayProps) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-4">
        <div className="space-y-1">
          <p className="text-sm text-muted-foreground">Best OA</p>
          <p className="text-2xl font-bold">{trainResult.best.oa.toFixed(3)}%</p>
        </div>
        <div className="space-y-1">
          <p className="text-sm text-muted-foreground">Best Epoch</p>
          <p className="text-2xl font-bold">{trainResult.best.epoch}</p>
        </div>
        <div className="space-y-1">
          <p className="text-sm text-muted-foreground">Best Kappa</p>
          <p className="text-2xl font-bold">{trainResult.best.kappa.toFixed(4)}</p>
        </div>
      </div>

      <div className="space-y-2">
        <h4 className="text-sm font-medium">Epoch Metrics</h4>
        <div className="rounded-md border">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b bg-muted/50">
                <th className="p-2 text-left">Epoch</th>
                <th className="p-2 text-left">OA</th>
                <th className="p-2 text-left">Kappa</th>
                <th className="p-2 text-left">F1</th>
                <th className="p-2 text-left">Precision</th>
                <th className="p-2 text-left">Recall</th>
                <th className="p-2 text-left">Loss</th>
              </tr>
            </thead>
            <tbody>
              {trainResult.metrics.map((metric) => (
                <tr key={metric.epoch} className="border-b">
                  <td className="p-2">{metric.epoch}</td>
                  <td className="p-2">{metric.oa.toFixed(3)}</td>
                  <td className="p-2">{metric.kappa.toFixed(4)}</td>
                  <td className="p-2">{metric.f1.toFixed(4)}</td>
                  <td className="p-2">{metric.precision.toFixed(4)}</td>
                  <td className="p-2">{metric.recall.toFixed(4)}</td>
                  <td className="p-2">{metric.loss.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
