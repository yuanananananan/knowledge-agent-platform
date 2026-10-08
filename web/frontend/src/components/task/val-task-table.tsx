import { useState } from 'react';
import {
  flexRender,
  getCoreRowModel,
  useReactTable,
  type RowSelectionState,
} from '@tanstack/react-table';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '../ui/table';
import { createColumns } from '@/components/task/val-task-column';
import { useDeleteValJob, useValTasks } from '@/hooks/val';
import { useNavigate } from 'react-router';
import { Button } from '../ui/button';
import { CheckCircle2 } from 'lucide-react';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '../ui/dialog';
import { useAuthStore } from '@/stores/auth';
import type { EffectivenessAnalysisData } from '@/types/analysis';
import CreateEffectivenessAnalysisDialog from './create-effectiveness-analysis-dialog';

export default function ValTaskTable() {
  const { data: valTasks = [], isLoading } = useValTasks();
  const { mutate: deleteValJob } = useDeleteValJob();
  const navigate = useNavigate();
  const user = useAuthStore((state) => state.user);

  const [rowSelection, setRowSelection] = useState<RowSelectionState>({});
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [createdAnalysis, setCreatedAnalysis] = useState<EffectivenessAnalysisData | null>(null);
  const [showResultsDialog, setShowResultsDialog] = useState(false);

  const handleViewDetail = (jobId: string) => {
    navigate(`/task/${jobId}`);
  };

  const handleDelete = (jobId: string) => {
    const ok = window.confirm('确定要删除这个测试任务吗？');
    if (!ok) return;
    deleteValJob(jobId);
  };

  const columns = createColumns(handleViewDetail, handleDelete);

  const table = useReactTable({
    data: valTasks,
    columns,
    getCoreRowModel: getCoreRowModel(),
    onRowSelectionChange: setRowSelection,
    state: {
      rowSelection,
    },
    enableRowSelection: true,
  });

  const selectedRows = table.getSelectedRowModel().rows;
  const selectedTasks = selectedRows.map((row) => row.original);
  const selectedCount = selectedTasks.length;

  return (
    <div className="space-y-4">
      {selectedCount > 0 && (
        <div className="flex items-center justify-between bg-muted/50 p-4 rounded-lg">
          <div className="text-sm text-muted-foreground">
            已选择 <span className="font-semibold text-foreground">{selectedCount}</span> 个测试任务
            <span className="ml-2 text-xs">（有效性分析必须正好选择 2 个）</span>
          </div>
          <Button onClick={() => setCreateDialogOpen(true)} disabled={selectedCount !== 2}>
            <CheckCircle2 className="mr-2 h-4 w-4" />
            创建有效性分析
          </Button>
        </div>
      )}

      <div className="rounded-md border">
        <Table className="table-fixed w-full">
          <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <TableHead key={header.id} style={{ width: header.column.getSize() }}>
                    {header.isPlaceholder
                      ? null
                      : flexRender(header.column.columnDef.header, header.getContext())}
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
      </div>

      {user && (
        <CreateEffectivenessAnalysisDialog
          open={createDialogOpen}
          onOpenChange={setCreateDialogOpen}
          selectedTasks={selectedTasks}
          userId={user.id}
          onSuccess={(record) => {
            setCreatedAnalysis(record);
            setShowResultsDialog(true);
          }}
        />
      )}

      <Dialog open={showResultsDialog} onOpenChange={setShowResultsDialog}>
        <DialogContent className="max-w-[70vw] sm:max-w-[70vw] w-full max-h-[80vh] overflow-hidden flex flex-col">
          <DialogHeader>
            <DialogTitle>有效性分析结果</DialogTitle>
            <DialogDescription>{createdAnalysis?.analysisName}</DialogDescription>
          </DialogHeader>

          <div className="flex-1 overflow-y-auto">
            {createdAnalysis?.result && (
              <div className="rounded-md border">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b bg-muted/50">
                      <th className="p-3 text-left">角色</th>
                      <th className="p-3 text-left">模型</th>
                      <th className="p-3 text-left">Backbone</th>
                      <th className="p-3 text-left">数据量(%)</th>
                      <th className="p-3 text-left">OA</th>
                      <th className="p-3 text-left">Precision</th>
                      <th className="p-3 text-left">Recall</th>
                      <th className="p-3 text-left">F1</th>
                      <th className="p-3 text-left">Kappa</th>
                      <th className="p-3 text-left">Loss</th>
                      <th className="p-3 text-left">提升率</th>
                      <th className="p-3 text-left">每样本提升率</th>
                    </tr>
                  </thead>
                  <tbody>
                    {createdAnalysis.result.rows.map((row) => (
                      <tr key={row.modelId} className="border-b">
                        <td className="p-3">{row.role}</td>
                        <td className="p-3">{row.modelName}</td>
                        <td className="p-3">{row.backbone}</td>
                        <td className="p-3">{row.dataRatio ?? '-'}</td>
                        <td className="p-3">{row.oa ?? '-'}</td>
                        <td className="p-3">{row.precision ?? '-'}</td>
                        <td className="p-3">{row.recall ?? '-'}</td>
                        <td className="p-3">{row.f1 ?? '-'}</td>
                        <td className="p-3">{row.kappa ?? '-'}</td>
                        <td className="p-3">{row.loss ?? '-'}</td>
                        <td className="p-3">{row.upliftRate ?? '-'}</td>
                        <td className="p-3">{row.perSampleGain ?? '-'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}