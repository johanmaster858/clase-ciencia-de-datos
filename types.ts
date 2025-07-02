
export interface HistogramData {
    name: string;
    value: number;
}

export interface ScatterData {
    x: number;
    y: number;
}

export interface ScatterDataWithColor extends ScatterData {
    z: number;
}

export interface ClusterData {
    x: number;
    y: number;
    cluster: number;
}

export interface CorrelationMatrix {
    labels: string[];
    values: number[][];
}
