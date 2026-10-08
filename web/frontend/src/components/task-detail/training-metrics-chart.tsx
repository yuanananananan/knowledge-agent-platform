import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  ChartLegend,
  ChartLegendContent,
} from '../ui/chart';
import { Line, LineChart, CartesianGrid, XAxis, YAxis } from 'recharts';
import type { Metric } from '@/types/task';

interface TrainingMetricsChartProps {
  metrics: Metric[];
}

export function TrainingMetricsChart({ metrics }: TrainingMetricsChartProps) {
  const chartData = metrics.map((metric) => ({
    epoch: metric.epoch,
    oa: metric.oa,
    kappa: metric.kappa,
    f1: metric.f1,
    precision: metric.precision,
    recall: metric.recall,
    loss: metric.loss,
  }));

  const accuracyConfig = {
    oa: {
      label: 'OA',
      color: 'var(--chart-1)',
    },
    kappa: {
      label: 'Kappa',
      color: 'var(--chart-2)',
    },
    f1: {
      label: 'F1',
      color: 'var(--chart-3)',
    },
  };

  const classificationConfig = {
    precision: {
      label: 'Precision',
      color: 'var(--chart-4)',
    },
    recall: {
      label: 'Recall',
      color: 'var(--chart-5)',
    },
  };

  const lossConfig = {
    loss: {
      label: 'Loss',
      color: 'var(--chart-1)',
    },
  };

  return (
    <Tabs defaultValue="accuracy" className="space-y-4">
      <TabsList>
        <TabsTrigger value="accuracy">准确率指标</TabsTrigger>
        <TabsTrigger value="classification">分类指标</TabsTrigger>
        <TabsTrigger value="loss">损失曲线</TabsTrigger>
      </TabsList>

      <TabsContent value="accuracy" className="m-0">
        <ChartContainer config={accuracyConfig} className="h-[400px] w-full">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="epoch"
              label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
            />
            <YAxis label={{ value: 'Score', angle: -90, position: 'insideLeft' }} />
            <ChartTooltip content={<ChartTooltipContent />} />
            <ChartLegend content={<ChartLegendContent />} />
            <Line
              type="monotone"
              dataKey="oa"
              stroke="var(--color-oa)"
              strokeWidth={2}
              dot={{ r: 4 }}
            />
            <Line
              type="monotone"
              dataKey="kappa"
              stroke="var(--color-kappa)"
              strokeWidth={2}
              dot={{ r: 4 }}
            />
            <Line
              type="monotone"
              dataKey="f1"
              stroke="var(--color-f1)"
              strokeWidth={2}
              dot={{ r: 4 }}
            />
          </LineChart>
        </ChartContainer>
      </TabsContent>

      <TabsContent value="classification" className="m-0">
        <ChartContainer config={classificationConfig} className="h-[400px] w-full">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="epoch"
              label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
            />
            <YAxis label={{ value: 'Score', angle: -90, position: 'insideLeft' }} />
            <ChartTooltip content={<ChartTooltipContent />} />
            <ChartLegend content={<ChartLegendContent />} />
            <Line
              type="monotone"
              dataKey="precision"
              stroke="var(--color-precision)"
              strokeWidth={2}
              dot={{ r: 4 }}
            />
            <Line
              type="monotone"
              dataKey="recall"
              stroke="var(--color-recall)"
              strokeWidth={2}
              dot={{ r: 4 }}
            />
          </LineChart>
        </ChartContainer>
      </TabsContent>

      <TabsContent value="loss" className="m-0">
        <ChartContainer config={lossConfig} className="h-[400px] w-full">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="epoch"
              label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
            />
            <YAxis label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} />
            <ChartTooltip content={<ChartTooltipContent />} />
            <ChartLegend content={<ChartLegendContent />} />
            <Line
              type="monotone"
              dataKey="loss"
              stroke="var(--color-loss)"
              strokeWidth={2}
              dot={{ r: 4 }}
            />
          </LineChart>
        </ChartContainer>
      </TabsContent>
    </Tabs>
  );
}
