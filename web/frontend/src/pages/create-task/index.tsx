import TrainTaskForm from '@/components/task/form/train-task-form';
import NasSearchForm from '@/components/task/form/nas-search-form';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

export default function CreateTaskPage() {
  return (
    <div className="container mx-auto max-w-5xl space-y-6 p-6">
      <div className="space-y-2">
        <h1 className="text-xl font-semibold">创建任务</h1>
        <p className="text-sm text-muted-foreground">
          可以创建训练任务，也可以先创建 NAS 主干搜索任务。
        </p>
      </div>

      <Tabs defaultValue="train-task">
        <TabsList>
          <TabsTrigger value="train-task">创建训练任务</TabsTrigger>
          <TabsTrigger value="nas-search">创建NAS主干搜索任务</TabsTrigger>
        </TabsList>

        <TabsContent value="train-task">
          <TrainTaskForm />
        </TabsContent>

        <TabsContent value="nas-search">
          <NasSearchForm />
        </TabsContent>
      </Tabs>
    </div>
  );
}