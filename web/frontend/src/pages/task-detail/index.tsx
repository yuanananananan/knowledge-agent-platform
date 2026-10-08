import { useParams, useNavigate } from 'react-router';
import { Button } from '@/components/ui/button';
import { ArrowLeft } from 'lucide-react';
import OnlineTrainLog from '@/components/task-detail/online-train-log';
import OfflineTrainResult from '@/components/task-detail/offline-train-result';
import { useTrainTasks } from '@/hooks/train';
import { useValTasks } from '@/hooks/val';
import { useMemo } from 'react';
import OfflineValResult from '@/components/task-detail/offline-val-result';

export default function TaskDetailPage() {
  const { job_id: jobId } = useParams<{ job_id: string }>();
  const { data: trainTasks = [] } = useTrainTasks();
  const { data: valTasks = [] } = useValTasks();
  const navigate = useNavigate();
  const trainTask = useMemo(() => {
    return trainTasks.find((task) => task.jobId === jobId);
  }, [trainTasks, jobId]);
  const valTask = useMemo(() => {
    return valTasks.find((task) => task.valJobId === jobId);
  }, [valTasks, jobId]);

  if (!trainTask && !valTask) {
    return null;
  }

  return (
    <div className="container mx-auto max-w-7xl p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => {
              navigate(-1);
            }}
          >
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <span className="px-2">任务详情</span>
          <span className="text-muted-foreground">{jobId}</span>
        </h1>
      </div>
      {valTask && <OfflineValResult jobId={jobId!}></OfflineValResult>}
      {trainTask?.status === 'completed' && (
        <OfflineTrainResult jobId={jobId!}></OfflineTrainResult>
      )}
      {trainTask?.status === 'running' && <OnlineTrainLog jobId={jobId!}></OnlineTrainLog>}
    </div>
  );
}
