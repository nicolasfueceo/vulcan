'use client'

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts'

interface GenerationData {
  generation: number
  total_features: number
  successful_features: number
  avg_score: number
  best_score: number
  action_taken: string
  population_size: number
}

interface GenerationChartProps {
  data: GenerationData[]
}

export function GenerationChart({ data }: GenerationChartProps) {
  if (!data.length) {
    return (
      <div className="flex items-center justify-center h-64 text-slate-500">
        <div className="text-center">
          <div className="text-2xl mb-2">📊</div>
          <p>No generation data available</p>
          <p className="text-sm">Start an experiment to see performance over time</p>
        </div>
      </div>
    )
  }

  // Prepare data for charts
  const chartData = data.map(gen => ({
    generation: gen.generation,
    avgScore: gen.avg_score,
    bestScore: gen.best_score,
    successRate: (gen.successful_features / gen.total_features) * 100,
    totalFeatures: gen.total_features,
    successfulFeatures: gen.successful_features,
    actionTaken: gen.action_taken,
    populationSize: gen.population_size
  }))

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white dark:bg-slate-800 p-3 border rounded-lg shadow-lg">
          <p className="font-semibold">Generation {label}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} style={{ color: entry.color }}>
              {entry.name}: {entry.value.toFixed(4)}
            </p>
          ))}
          {payload[0]?.payload?.actionTaken && (
            <p className="text-sm text-slate-600 mt-1">
              Action: {payload[0].payload.actionTaken.replace('_', ' ')}
            </p>
          )}
        </div>
      )
    }
    return null
  }

  return (
    <div className="space-y-8">
      {/* Score Evolution */}
      <div>
        <h3 className="text-lg font-semibold mb-4">Score Evolution</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="generation" 
              label={{ value: 'Generation', position: 'insideBottom', offset: -10 }}
            />
            <YAxis 
              label={{ value: 'Score', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="avgScore" 
              stroke="#8884d8" 
              strokeWidth={2}
              name="Average Score"
              dot={{ fill: '#8884d8' }}
            />
            <Line 
              type="monotone" 
              dataKey="bestScore" 
              stroke="#82ca9d" 
              strokeWidth={2}
              name="Best Score"
              dot={{ fill: '#82ca9d' }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Success Rate */}
      <div>
        <h3 className="text-lg font-semibold mb-4">Success Rate</h3>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="generation"
              label={{ value: 'Generation', position: 'insideBottom', offset: -10 }}
            />
            <YAxis 
              label={{ value: 'Success Rate (%)', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip 
              formatter={(value: any) => [`${value.toFixed(1)}%`, 'Success Rate']}
              labelFormatter={(label) => `Generation ${label}`}
            />
            <Bar 
              dataKey="successRate" 
              fill="#fbbf24"
              name="Success Rate"
            />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Feature Generation */}
      <div>
        <h3 className="text-lg font-semibold mb-4">Feature Generation</h3>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="generation"
              label={{ value: 'Generation', position: 'insideBottom', offset: -10 }}
            />
            <YAxis 
              label={{ value: 'Features', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip />
            <Legend />
            <Bar 
              dataKey="totalFeatures" 
              fill="#6366f1"
              name="Total Features"
            />
            <Bar 
              dataKey="successfulFeatures" 
              fill="#10b981"
              name="Successful Features"
            />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Statistics Table */}
      <div>
        <h3 className="text-lg font-semibold mb-4">Generation Statistics</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse border border-slate-300 dark:border-slate-600">
            <thead>
              <tr className="bg-slate-50 dark:bg-slate-800">
                <th className="border border-slate-300 dark:border-slate-600 px-3 py-2 text-left">Gen</th>
                <th className="border border-slate-300 dark:border-slate-600 px-3 py-2 text-left">Action</th>
                <th className="border border-slate-300 dark:border-slate-600 px-3 py-2 text-right">Total</th>
                <th className="border border-slate-300 dark:border-slate-600 px-3 py-2 text-right">Success</th>
                <th className="border border-slate-300 dark:border-slate-600 px-3 py-2 text-right">Avg Score</th>
                <th className="border border-slate-300 dark:border-slate-600 px-3 py-2 text-right">Best Score</th>
                <th className="border border-slate-300 dark:border-slate-600 px-3 py-2 text-right">Pop Size</th>
              </tr>
            </thead>
            <tbody>
              {data.map((gen, index) => (
                <tr key={gen.generation} className={index % 2 === 0 ? 'bg-white dark:bg-slate-900' : 'bg-slate-50 dark:bg-slate-800'}>
                  <td className="border border-slate-300 dark:border-slate-600 px-3 py-2">{gen.generation}</td>
                  <td className="border border-slate-300 dark:border-slate-600 px-3 py-2">
                    <span className={`px-2 py-1 rounded text-xs ${
                      gen.action_taken === 'generate_new' 
                        ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
                        : 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200'
                    }`}>
                      {gen.action_taken.replace('_', ' ')}
                    </span>
                  </td>
                  <td className="border border-slate-300 dark:border-slate-600 px-3 py-2 text-right">{gen.total_features}</td>
                  <td className="border border-slate-300 dark:border-slate-600 px-3 py-2 text-right">{gen.successful_features}</td>
                  <td className="border border-slate-300 dark:border-slate-600 px-3 py-2 text-right">{gen.avg_score.toFixed(4)}</td>
                  <td className="border border-slate-300 dark:border-slate-600 px-3 py-2 text-right">{gen.best_score.toFixed(4)}</td>
                  <td className="border border-slate-300 dark:border-slate-600 px-3 py-2 text-right">{gen.population_size}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
} 