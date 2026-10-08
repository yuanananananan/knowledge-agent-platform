import { useState } from 'react';
import {
  flexRender,
  getCoreRowModel,
  useReactTable,
} from '@tanstack/react-table';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '../ui/table';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '../ui/dialog';
import { createNasSearchColumns } from './nas-search-task-column';
import {
  useDeleteNasSearchJob,
  useNasSearchJobs,
  useNasSearchLogs,
} from '@/hooks/nas';
import type { NasSearchJobData } from '@/types/nas';

export default function NasSearchTaskTable() {
  const { data: nasSearchJobs = [], isLoading } = useNasSearchJobs();
  const { mutate: deleteNasSearchJob } = useDeleteNasSearchJob();

  const [logDialogOpen, setLogDialogOpen] = useState(false);
  const [selectedTask, setSelectedTask] = useState<NasSearchJobData | null>(null);

  const { data: logsData, isLoading: logsLoading } = useNasSearchLogs(
    selectedTask?.jobId,
  );

  const handleDelete = (task: NasSearchJobData) => {
    const ok = window.confirm('确定要删除这个NAS搜索任务吗？');

    if (!ok) {
      return;
    }

    deleteNasSearchJob(task.id);
  };

  const handleViewLogs = (task: NasSearchJobData) => {
    setSelectedTask(task);
    setLogDialogOpen(true);
  };

  const columns = createNasSearchColumns({
    onDelete: handleDelete,
    onViewLogs: handleViewLogs,
  });

  const table = useReactTable({
    data: nasSearchJobs,
    columns,
    getCoreRowModel: getCoreRowModel(),
  });

  return (
    <div className="space-y-4">
      <div className="rounded-md border">
        <Table className="w-full table-fixed">
          <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <TableHead
                    key={header.id}
                    style={{ width: header.column.getSize() }}
                  >
                    {header.isPlaceholder
                      ? null
                      : flexRender(
                          header.column.columnDef.header,
                          header.getContext(),
                        )}
                  </TableHead>
                ))}
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
            ) : table.getRowModel().rows.length ? (
              table.getRowModel().rows.map((row) => (
                <TableRow key={row.id}>
                  {row.getVisibleCells().map((cell) => (
                    <TableCell
                      key={cell.id}
                      style={{ width: cell.column.getSize() }}
                    >
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext(),
                      )}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell colSpan={columns.length} className="h-24 text-center">
                  暂无NAS主干搜索任务
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>

      <Dialog open={logDialogOpen} onOpenChange={setLogDialogOpen}>
        <DialogContent className="max-h-[80vh] max-w-[80vw] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>NAS搜索日志</DialogTitle>
          </DialogHeader>

          {logsLoading ? (
            <div className="text-sm text-muted-foreground">日志加载中...</div>
          ) : (
            <pre className="whitespace-pre-wrap rounded-md bg-muted p-4 text-xs">
              {logsData?.log || '暂无日志'}
            </pre>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}