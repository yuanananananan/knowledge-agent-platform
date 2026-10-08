import { useState } from 'react';
import { useNavigate } from 'react-router';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { useDatasets } from '@/hooks/dataset/use-datasets';
import { useStartNasSearch } from '@/hooks/nas';
import { useAuthStore } from '@/stores/auth';

export default function NasSearchForm() {
  const navigate = useNavigate();
  const user = useAuthStore((state) => state.user);

  const { data: datasets = [] } = useDatasets();
  const { mutate: startNasSearch, isPending } = useStartNasSearch();

  const [datasetId, setDatasetId] = useState<number>(0);
  const [modality, setModality] = useState<string>('sar');
  const [description, setDescription] = useState('');

  const [epochs, setEpochs] = useState(30);
  const [batchSize, setBatchSize] = useState(2);
  const [initChannels, setInitChannels] = useState(16);
  const [layers, setLayers] = useState(8);
  const [learningRate, setLearningRate] = useState(0.025);
  const [learningRateMin, setLearningRateMin] = useState(0.001);
  const [momentum, setMomentum] = useState(0.9);
  const [weightDecay, setWeightDecay] = useState(0.0003);
  const [archLearningRate, setArchLearningRate] = useState(0.0003);
  const [archWeightDecay, setArchWeightDecay] = useState(0.001);
  const [trainPortion, setTrainPortion] = useState(0.5);
  const [gradClip, setGradClip] = useState(5.0);

  const handleSubmit = () => {
    if (!user || !datasetId) {
      return;
    }

    startNasSearch(
      {
        userId: user.id,
        datasetId,
        modality,
        description,
        config: {
          epochs,
          batch_size: batchSize,
          init_channels: initChannels,
          layers,
          learning_rate: learningRate,
          learning_rate_min: learningRateMin,
          momentum,
          weight_decay: weightDecay,
          arch_learning_rate: archLearningRate,
          arch_weight_decay: archWeightDecay,
          train_portion: trainPortion,
          grad_clip: gradClip,
          unrolled: true,
        },
      },
      {
        onSuccess: () => {
          navigate('/task');
        },
      },
    );
  };

  return (
    <div className="space-y-6 rounded-md border p-6">
      <div>
        <h2 className="text-lg font-medium">创建NAS主干搜索任务</h2>
        <p className="text-sm text-muted-foreground">
          调用 ml/NAS/train_search.py 搜索最优主干结构，搜索完成后自动注册为 Backbone。
        </p>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <label className="text-sm font-medium">数据集</label>

          <Select
            value={datasetId ? String(datasetId) : ''}
            onValueChange={(value) => {
              const id = Number(value);
              setDatasetId(id);

              const dataset = datasets.find((item) => item.id === id);
              if (dataset?.modality) {
                setModality(dataset.modality.toLowerCase());
              }
            }}
          >
            <SelectTrigger>
              <SelectValue placeholder="请选择数据集" />
            </SelectTrigger>

            <SelectContent>
              {datasets.map((dataset) => (
                <SelectItem key={dataset.id} value={String(dataset.id)}>
                  {dataset.name} / {dataset.modality}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">模态</label>

          <Select value={modality} onValueChange={setModality}>
            <SelectTrigger>
              <SelectValue placeholder="请选择模态" />
            </SelectTrigger>

            <SelectContent>
              <SelectItem value="sar">SAR</SelectItem>
              <SelectItem value="rd">RD</SelectItem>
              <SelectItem value="mon">MON</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium">任务描述</label>

        <Textarea
          value={description}
          onChange={(event) => setDescription(event.target.value)}
          placeholder="例如：SAR模态NAS主干搜索"
        />
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div className="space-y-2">
          <label className="text-sm font-medium">epochs</label>
          <Input
            type="number"
            value={epochs}
            onChange={(event) => setEpochs(Number(event.target.value))}
          />
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">batch_size</label>
          <Input
            type="number"
            value={batchSize}
            onChange={(event) => setBatchSize(Number(event.target.value))}
          />
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">init_channels</label>
          <Input
            type="number"
            value={initChannels}
            onChange={(event) => setInitChannels(Number(event.target.value))}
          />
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">layers</label>
          <Input
            type="number"
            value={layers}
            onChange={(event) => setLayers(Number(event.target.value))}
          />
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">learning_rate</label>
          <Input
            type="number"
            step="0.0001"
            value={learningRate}
            onChange={(event) => setLearningRate(Number(event.target.value))}
          />
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">learning_rate_min</label>
          <Input
            type="number"
            step="0.0001"
            value={learningRateMin}
            onChange={(event) => setLearningRateMin(Number(event.target.value))}
          />
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">momentum</label>
          <Input
            type="number"
            step="0.1"
            value={momentum}
            onChange={(event) => setMomentum(Number(event.target.value))}
          />
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">weight_decay</label>
          <Input
            type="number"
            step="0.0001"
            value={weightDecay}
            onChange={(event) => setWeightDecay(Number(event.target.value))}
          />
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">arch_learning_rate</label>
          <Input
            type="number"
            step="0.0001"
            value={archLearningRate}
            onChange={(event) => setArchLearningRate(Number(event.target.value))}
          />
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">arch_weight_decay</label>
          <Input
            type="number"
            step="0.0001"
            value={archWeightDecay}
            onChange={(event) => setArchWeightDecay(Number(event.target.value))}
          />
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">train_portion</label>
          <Input
            type="number"
            step="0.1"
            value={trainPortion}
            onChange={(event) => setTrainPortion(Number(event.target.value))}
          />
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">grad_clip</label>
          <Input
            type="number"
            step="0.1"
            value={gradClip}
            onChange={(event) => setGradClip(Number(event.target.value))}
          />
        </div>
      </div>

      <div className="flex justify-end">
        <Button onClick={handleSubmit} disabled={!user || !datasetId || isPending}>
          {isPending ? '启动中...' : '开始NAS搜索'}
        </Button>
      </div>
    </div>
  );
}
