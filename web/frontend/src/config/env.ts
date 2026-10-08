// API配置类
class ApiConfig {
  public readonly baseUrl: string;
  public readonly timeout: number;
  public readonly isDevelopment: boolean;

  constructor() {
    // 验证必需的环境变量
    this.validateRequiredEnvVars();

    // 初始化API配置
    this.baseUrl = import.meta.env.VITE_API_BASE_URL;
    this.timeout = this.getTimeout();
    this.isDevelopment = import.meta.env.MODE === 'development';

    // 验证URL格式
    this.validateApiUrl();
  }

  /**
   * 验证必需的环境变量
   */
  private validateRequiredEnvVars(): void {
    if (!import.meta.env.VITE_API_BASE_URL) {
      throw new Error('缺少必需的环境变量: VITE_API_BASE_URL');
    }
  }

  /**
   * 获取API超时配置
   */
  private getTimeout(): number {
    const timeoutStr = import.meta.env.VITE_API_TIMEOUT;
    if (!timeoutStr) return 30000; // 默认30秒

    const timeout = Number(timeoutStr);
    return isNaN(timeout) ? 30000 : timeout;
  }

  /**
   * 验证API URL格式
   */
  private validateApiUrl(): void {
    try {
      new URL(this.baseUrl);
    } catch {
      throw new Error(`API Base URL 格式无效: ${this.baseUrl}`);
    }
  }

  /**
   * 获取完整的API URL
   */
  public getUrl(path: string = ''): string {
    const baseUrl = this.baseUrl.replace(/\/$/, ''); // 移除末尾斜杠
    const cleanPath = path.replace(/^\//, ''); // 移除开头斜杠
    return cleanPath ? `${baseUrl}/${cleanPath}` : baseUrl;
  }
}

// 创建API配置实例
export const apiConfig = new ApiConfig();

// 默认导出
export default apiConfig;
