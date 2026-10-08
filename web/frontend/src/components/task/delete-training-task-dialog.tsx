import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { useDeleteTrainingTask } from '@/hooks/train/use-delete-training-task';
import type { TrainTaskData } from '@/types/task';

interface DeleteTrainingTaskDialogProps {
  open: boolean;
  onOpenChange: (_open: boolean) => void;
  task: TrainTaskData | null;
}

export function DeleteTrainingTaskDialog({
  open,
  onOpenChange,
  task,
}: DeleteTrainingTaskDialogProps) {
  const { mutate: deleteTask, isPending } = useDeleteTrainingTask();

  const handleDelete = async () => {
    if (!task) return;

    deleteTask(task.jobId, {
      onSuccess: () => {
        onOpenChange(false);
      },
    });
  };

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle className="text-base font-semibold">删除训练任务</AlertDialogTitle>
          <AlertDialogDescription>
            确定要删除训练任务 <strong>{task?.jobId}</strong> 吗？此操作无法撤销。
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel disabled={isPending}>取消</AlertDialogCancel>
          <AlertDialogAction onClick={handleDelete} disabled={isPending}>
            {isPending ? '删除中...' : '确认删除'}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
