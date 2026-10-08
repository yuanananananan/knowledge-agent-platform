import TrainTaskTable from '@/components/task/train-task-table';
import ValTaskTable from '@/components/task/val-task-table';
import EffectivenessAnalysisTable from '@/components/task/effectiveness-analysis-table';
import NasSearchTaskTable from '@/components/task/nas-search-task-table';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

export default function TaskPage() {
  return (
    <div className="container mx-auto max-w-7xl space-y-6 p-6">
      <div className="flex items-center justify-between">
        <div className="space-y-2">
          <h1 className="text-xl font-semibold">任务列表</h1>
          <p className="text-sm text-muted-foreground sm:text-base">
            管理训练、NAS主干搜索、测试和有效性分析任务
          </p>
        </div>
      </div>

      <Tabs defaultValue="train-task">
        <TabsList>
          <TabsTrigger value="train-task">训练任务</TabsTrigger>
          <TabsTrigger value="nas-search-task">NAS主干搜索</TabsTrigger>
          <TabsTrigger value="val-task">测试任务</TabsTrigger>
          <TabsTrigger value="analysis-task">分析记录</TabsTrigger>
        </TabsList>

        <TabsContent value="train-task">
          <TrainTaskTable />
        </TabsContent>

        <TabsContent value="nas-search-task">
          <NasSearchTaskTable />
        </TabsContent>

        <TabsContent value="val-task">
          <ValTaskTable />
        </TabsContent>

        <TabsContent value="analysis-task">
          <EffectivenessAnalysisTable />
        </TabsContent>
      </Tabs>
    </div>
  );
}