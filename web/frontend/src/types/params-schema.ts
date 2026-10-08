export type Property = {
  type: 'string' | 'number' | 'boolean'; // 参数的类型
  description: string;
  default: string | number | boolean; // 默认值
  maximum?: number; // 数值范围
  minimum?: number;
};

export type ParamsSchema = {
  type: string; // 这个schema的类型，可以是任意的命名
  additionalProperties: boolean; //false表示禁止添加未定义的参数字段
  required: string[]; // 必填的参数
  properties: Property;
};
