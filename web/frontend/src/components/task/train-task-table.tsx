import { flexRender, getCoreRowModel, useReactTable } from '@tanstack/react-table';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '../ui/table';
import { createColumns } from '@/components/task/train-task-column';
import { useNavigate } from 'react-router';
import { useTrainTasks } from '@/hooks/train';
import { useStartValidation } from '@/hooks/val';
import { useAuthStore } from '@/stores/auth';
import { DeleteTrainingTaskDialog } from './delete-training-task-dialog';
import { useState } from 'react';
import type { TrainTaskData } from '@/types/task';

export default function TrainTaskTable() {
  const { data: trainTasks = [], isLoading } = useTrainTasks();
  const { mutate: startValidation } = useStartValidation();
  const user = useAuthStore((state) => state.user);
  const navigate = useNavigate();
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [taskToDelete, setTaskToDelete] = useState<TrainTaskData | null>(null);

  const handleViewDetail = (jobId: string) => {
    navigate(`/task/${jobId}`);
  };

  const handleStartVal = (jobId: string) => {
    startValidation({
      userId: user.id,
      trainJobId: jobId,
    });
  };

  const handleDelete = (task: TrainTaskData) => {
    setTaskToDelete(task);
    setDeleteDialogOpen(true);
  };

  const columns = createColumns({
    onViewDetail: handleViewDetail,
    onStartVal: handleStartVal,
    onDelete: handleDelete,
  });

  const table = useReactTable({
    data: trainTasks,
    columns,
    getCoreRowModel: getCoreRowModel(),
  });

  return (
    <div className="rounded-md border">
      <Table className="table-fixed w-full">
        <TableHeader>
          {table.getHeaderGroups().map((headerGroup) => (
            <TableRow key={headerGroup.id}>
              {headerGroup.headers.map((header) => {
                return (
                  <TableHead key={header.id} style={{ width: header.column.getSize() }}>
                    {header.isPlaceholder
                      ? null
                      : flexRender(header.column.columnDef.header, header.getContext())}
                  </TableHead>
                );
              })}
            </TableRow>
          ))}
        </TableHeader>
        <TableBody>
          {isLoading ? (
            <TableRow>
              <TableCell colSpan={columns.length} className="h-24 text-center">
                加载中...
              </TableCell>
            </TableRow>
          ) : table.getRowModel().rows?.length ? (
            table.getRowModel().rows.map((row) => (
              <TableRow key={row.id} data-state={row.getIsSelected() && 'selected'}>
                {row.getVisibleCells().map((cell) => (
                  <TableCell key={cell.id} style={{ width: cell.column.getSize() }}>
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </TableCell>
                ))}
              </TableRow>
            ))
          ) : (
            <TableRow>
              <TableCell colSpan={columns.length} className="h-24 text-center">
                没有找到任务数据
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
      <DeleteTrainingTaskDialog
        open={deleteDialogOpen}
        onOpenChange={setDeleteDialogOpen}
        task={taskToDelete}
      />
    </div>
  );
}
