'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { EvolutionData } from '@/types/vulcan';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface ExperimentChartsProps {
  data: EvolutionData;
}

export function ExperimentCharts({ data }: ExperimentChartsProps) {
  if (!data || !data.generation_history || data.generation_history.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Performance Charts</CardTitle>
        </CardHeader>
        <CardContent>
          <p>No generation data available to display charts.</p>
        </CardContent>
      </Card>
    );
  }

  const scoreData = data.generation_history.map(gen => ({
    name: `Gen ${gen.generation}`,
    'Best Score': gen.best_score.toFixed(4),
    'Avg Score': gen.avg_score.toFixed(4),
  }));

  const agentData = data.agent_stats
    ? Object.entries(data.agent_stats).map(([name, stats]) => ({
        name,
        'Times Chosen': stats.count,
        'Avg Reward': stats.count > 0 ? (stats.reward_sum / stats.count).toFixed(4) : 0,
      }))
    : [];

  return (
    <div className="grid gap-4 md:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle>Score Over Generations</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={scoreData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="Best Score" stroke="#8884d8" activeDot={{ r: 8 }} />
              <Line type="monotone" dataKey="Avg Score" stroke="#82ca9d" />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Agent Performance</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={agentData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="Times Chosen" fill="#8884d8" />
              <Bar dataKey="Avg Reward" fill="#82ca9d" />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
} 