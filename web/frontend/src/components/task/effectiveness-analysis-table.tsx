import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { useAuthStore } from '@/stores/auth';
import { useAnalysisRecords, useDeleteAnalysis } from '@/hooks/analysis';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import type { EffectivenessAnalysisData } from '@/types/analysis';

export default function EffectivenessAnalysisTable() {
  const user = useAuthStore((state) => state.user);
  const userId = user?.id ?? 0;

  const { data, isLoading } = useAnalysisRecords(userId, 1, 20);
  const { mutate: deleteAnalysis } = useDeleteAnalysis(userId);

  const [current, setCurrent] = useState<EffectivenessAnalysisData | null>(null);
  const [open, setOpen] = useState(false);

  if (!user) return null;

  return (
    <div className="space-y-4">
      <div className="rounded-md border">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b bg-muted/50">
              <th className="p-3 text-left">分析名称</th>
              <th className="p-3 text-left">Backbone</th>
              <th className="p-3 text-left">指标</th>
              <th className="p-3 text-left">状态</th>
              <th className="p-3 text-left">创建时间</th>
              <th className="p-3 text-left">操作</th>
            </tr>
          </thead>
          <tbody>
            {isLoading ? (
              <tr>
                <td className="p-3" colSpan={6}>
                  加载中...
                </td>
              </tr>
            ) : data?.items.length ? (
              data.items.map((item) => (
                <tr key={item.analysisId} className="border-b">
                  <td className="p-3">{item.analysisName}</td>
                  <td className="p-3">{item.backbone}</td>
                  <td className="p-3">{item.metricKey}</td>
                  <td className="p-3">{item.status}</td>
                  <td className="p-3">{item.createTime}</td>
                  <td className="p-3 space-x-2">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => {
                        setCurrent(item);
                        setOpen(true);
                      }}
                    >
                      查看
                    </Button>
                    <Button
                      size="sm"
                      variant="destructive"
                      onClick={() => deleteAnalysis(item.analysisId)}
                    >
                      删除
                    </Button>
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td className="p-3" colSpan={6}>
                  暂无分析记录
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="max-w-[70vw] sm:max-w-[70vw] w-full max-h-[80vh] overflow-hidden flex flex-col">
          <DialogHeader>
            <DialogTitle>{current?.analysisName}</DialogTitle>
          </DialogHeader>

          <div className="flex-1 overflow-y-auto">
            {current?.result && (
              <div className="rounded-md border">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b bg-muted/50">
                        <th className="p-3 text-left">角色</th>
                        <th className="p-3 text-left">模型</th>
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
                    {current.result.rows.map((row) => (
                      <tr key={row.modelId} className="border-b">
                        <td className="p-3">{row.role}</td>
                        <td className="p-3">{row.modelName}</td>
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