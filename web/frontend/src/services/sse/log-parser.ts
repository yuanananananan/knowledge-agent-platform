import type { Metric } from '@/types/task';

/**
 * Parse a training log line into a Metric object
 * Example log format:
 * [13:27:10] epoch:0, OA: 53.24%, kappa: 0.1970, Cls Acc: [99.58, 26.24, 0.0], P:0.4552, R:0.4194, F1:0.3493, loss:0.98
 */
export function parseLogToMetric(logLine: string): Metric | null {
  // Skip non-epoch logs (like "Start training" or "Training completed")
  if (!logLine.includes('epoch:')) {
    return null;
  }

  try {
    // Extract epoch number
    const epochMatch = logLine.match(/epoch:(\d+)/);
    const epoch = epochMatch ? parseInt(epochMatch[1], 10) : 0;

    // Extract OA (Overall Accuracy) - may be in format "53.24%" or just number
    const oaMatch = logLine.match(/OA:\s*([\d.]+)%?/);
    const oa = oaMatch ? parseFloat(oaMatch[1]) / 100 : 0;

    // Extract kappa
    const kappaMatch = logLine.match(/kappa:\s*([\d.]+)/i);
    const kappa = kappaMatch ? parseFloat(kappaMatch[1]) : 0;

    // Extract Class Accuracy array - format: [99.58, 26.24, 0.0]
    const clsAccMatch = logLine.match(/Cls Acc:\s*\[([\d., ]+)\]/i);
    const classAccuracy = clsAccMatch
      ? clsAccMatch[1].split(',').map((v) => parseFloat(v.trim()))
      : [];

    // Extract Precision (P)
    const precisionMatch = logLine.match(/P:([\d.]+)/);
    const precision = precisionMatch ? parseFloat(precisionMatch[1]) : 0;

    // Extract Recall (R)
    const recallMatch = logLine.match(/R:([\d.]+)/);
    const recall = recallMatch ? parseFloat(recallMatch[1]) : 0;

    // Extract F1
    const f1Match = logLine.match(/F1:([\d.]+)/);
    const f1 = f1Match ? parseFloat(f1Match[1]) : 0;

    // Extract loss - this is trainLoss, testLoss would need to come from result API
    const lossMatch = logLine.match(/loss:([\d.]+)/);
    const trainLoss = lossMatch ? parseFloat(lossMatch[1]) : 0;

    return {
      epoch,
      oa,
      kappa,
      classAccuracy,
      precision,
      recall,
      f1,
      loss: trainLoss,
    };
  } catch (error) {
    console.error('Failed to parse log line:', logLine, error);
    return null;
  }
}

/**
 * Check if a log line contains metric data (i.e., is an epoch log)
 */
export function isMetricLog(logLine: string): boolean {
  return logLine.includes('epoch:');
}
