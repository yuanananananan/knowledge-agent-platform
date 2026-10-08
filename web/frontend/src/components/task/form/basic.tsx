import { taskFormOpts, withForm } from '@/hooks/task/form';
import { FieldGroup } from '@/components/ui/field';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import type { Dataset } from '@/types/dataset';
import type { Classifier } from '@/types/classifier';
import type { Backbone } from '@/types/backbone';
import { Switch } from '@/components/ui/switch';

export const BasicField = withForm({
  ...taskFormOpts,
  props: {
    datasetItems: [] as Dataset[],
    classifierItems: [] as Classifier[],
    backboneItems: [] as Backbone[],
  },
  render: ({ form, datasetItems, classifierItems, backboneItems }) => (
    <Card className="rounded border-none">
      <CardHeader>
        <CardTitle>基本配置</CardTitle>
        <CardDescription>
          对训练的基本资源和结构进行配置，选择训练数据集、主干网络以及分类器
        </CardDescription>
      </CardHeader>

      <CardContent>
        <FieldGroup>
          <form.AppField name="datasetId">
            {(field) => (
              <field.SelectField
                title="数据集"
                description="选择用于训练的数据集"
                items={datasetItems}
              />
            )}
          </form.AppField>

          <form.AppField name="backboneId">
            {(field) => (
              <field.SelectField
                title="主干网络"
                description="选择主干网络，影响模型结构。选择 NAS 搜索主干时会自动启用 NAS 训练流程。"
                items={backboneItems}
              />
            )}
          </form.AppField>

          <form.AppField name="isGNN">
            {(field) => (
              <div className="flex items-center justify-between rounded-md border p-3 lg:w-2/3 xl:w-1/2">
                <div>
                  <div className="text-sm font-medium">是否采用嵌入知识</div>
                  <div className="text-xs text-muted-foreground">
                    开启后使用知识嵌入/GNN；关闭后只训练识别主干。
                  </div>
                </div>

                <Switch
                  checked={field.state.value}
                  onCheckedChange={field.handleChange}
                />
              </div>
            )}
          </form.AppField>

          <form.AppField name="classifierId">
            {(field) => (
              <field.SelectField
                title="分类器"
                description="选择网络模型的分类器"
                items={classifierItems}
              />
            )}
          </form.AppField>
        </FieldGroup>
      </CardContent>
    </Card>
  ),
});