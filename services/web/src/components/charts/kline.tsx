import React from 'react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend,
  ResponsiveContainer 
} from 'recharts';
import { useDataFrameStore } from '../../store/dataframe';
import { useConfigsStore } from '../../store/configs';
import * as animated from '../ui/animated';
import { Loader } from '../ui/loader';

export enum ChartType {
  CANDLE = 'candle_solid',
  AREA = 'area',
}

export enum AxisType {
  NORMAL = 'normal',
  PERCENTAGE = 'percentage',
}

interface KlineChartProps {
  type: ChartType;
  axis: AxisType;
  primary: any[];
  secondary: any[];
}

export function KlineChart(props: KlineChartProps): React.ReactElement {
  const getActiveConfig = useConfigsStore(state => state.getActiveConfig);
  const loading = useDataFrameStore(state => state.loading);
  const get = useDataFrameStore(state => state.get);

  const { symbol } = getActiveConfig();
  const data = get(symbol);

  const chartData = data.map(({ kline, prediction }) => ({
    timestamp: kline.time,
    open: kline.open,
    high: kline.high,
    low: kline.low,
    close: kline.close,
    volume: kline.volume,
    prediction: prediction
  }));

  return (
    <>
      <animated.Div className='w-full h-[92%]'>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={chartData}
            margin={{
              top: 20,
              right: 30,
              left: 20,
              bottom: 10
            }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200" />
            
            <XAxis 
              dataKey="timestamp"
              tickFormatter={(value) => new Date(value).toLocaleTimeString()}
              className="text-sm" 
            />
            
            <YAxis 
              yAxisId="price"
              domain={['auto', 'auto']}
              className="text-sm"
            />
            
            <YAxis 
              yAxisId="volume"
              orientation="right"
              domain={['auto', 'auto']}
              className="text-sm"
            />
            
            <Tooltip
              contentStyle={{ backgroundColor: 'white', border: '1px solid #ccc' }}
              labelFormatter={(value) => new Date(value).toLocaleString()}
            />
            
            <Legend />
            
            <Line
              yAxisId="price"
              type="monotone"
              dataKey="close"
              stroke="#2563eb"
              dot={false}
              name="Price"
            />
            
            <Line
              yAxisId="price"
              type="monotone"
              dataKey="prediction"
              stroke="#ef4444"
              strokeDasharray="5 5"
              dot={false}
              name="Prediction"
            />
            
            <Line
              yAxisId="volume"
              type="bar"
              dataKey="volume"
              stroke="#6b7280"
              fill="#6b7280"
              name="Volume"
            />
          </LineChart>
        </ResponsiveContainer>
      </animated.Div>
      <Loader className='absolute top-0 left-0' visible={loading} />
    </>
  );
}