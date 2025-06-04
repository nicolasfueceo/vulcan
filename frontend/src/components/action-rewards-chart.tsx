'use client'

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

interface ActionRewardsChartProps {
  data: {
    generate_new: number[]
    mutate_existing: number[]
  }
}

export function ActionRewardsChart({ data }: ActionRewardsChartProps) {
  if (!data.generate_new.length && !data.mutate_existing.length) {
    return (
      <div className="flex items-center justify-center h-64 text-slate-500">
        <div className="text-center">
          <div className="text-2xl mb-2">🎯</div>
          <p>No RL action data available</p>
          <p className="text-sm">Start an experiment to see action rewards</p>
        </div>
      </div>
    )
  }

  // Prepare data for chart
  const maxLength = Math.max(data.generate_new.length, data.mutate_existing.length)
  const chartData = Array.from({ length: maxLength }, (_, index) => ({
    step: index + 1,
    generateNew: data.generate_new[index] || null,
    mutateExisting: data.mutate_existing[index] || null
  }))

  // Calculate statistics
  const generateNewAvg = data.generate_new.length > 0 
    ? data.generate_new.reduce((sum, val) => sum + val, 0) / data.generate_new.length 
    : 0

  const mutateExistingAvg = data.mutate_existing.length > 0 
    ? data.mutate_existing.reduce((sum, val) => sum + val, 0) / data.mutate_existing.length 
    : 0

  const generateNewTrend = data.generate_new.length > 1
    ? data.generate_new[data.generate_new.length - 1] - data.generate_new[0]
    : 0

  const mutateExistingTrend = data.mutate_existing.length > 1
    ? data.mutate_existing[data.mutate_existing.length - 1] - data.mutate_existing[0]
    : 0

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white dark:bg-slate-800 p-3 border rounded-lg shadow-lg">
          <p className="font-semibold">Step {label}</p>
          {payload.map((entry: any, index: number) => (
            entry.value !== null && (
              <p key={index} style={{ color: entry.color }}>
                {entry.name}: {entry.value.toFixed(4)}
              </p>
            )
          ))}
        </div>
      )
    }
    return null
  }

  return (
    <div className="space-y-6">
      {/* Statistics Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="text-center p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
          <div className="text-lg font-bold text-blue-600">{generateNewAvg.toFixed(4)}</div>
          <div className="text-sm text-blue-600">Avg Generate New</div>
        </div>
        <div className="text-center p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
          <div className="text-lg font-bold text-purple-600">{mutateExistingAvg.toFixed(4)}</div>
          <div className="text-sm text-purple-600">Avg Mutate Existing</div>
        </div>
        <div className="text-center p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
          <div className={`text-lg font-bold ${generateNewTrend >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            {generateNewTrend >= 0 ? '+' : ''}{generateNewTrend.toFixed(4)}
          </div>
          <div className="text-sm text-green-600">Generate Trend</div>
        </div>
        <div className="text-center p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
          <div className={`text-lg font-bold ${mutateExistingTrend >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            {mutateExistingTrend >= 0 ? '+' : ''}{mutateExistingTrend.toFixed(4)}
          </div>
          <div className="text-sm text-orange-600">Mutate Trend</div>
        </div>
      </div>

      {/* Rewards Chart */}
      <div>
        <h3 className="text-lg font-semibold mb-4">Action Rewards Over Time</h3>
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="step" 
              label={{ value: 'Decision Step', position: 'insideBottom', offset: -10 }}
            />
            <YAxis 
              label={{ value: 'Reward', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="generateNew" 
              stroke="#3b82f6" 
              strokeWidth={2}
              name="Generate New"
              dot={{ fill: '#3b82f6', r: 4 }}
              connectNulls={false}
            />
            <Line 
              type="monotone" 
              dataKey="mutateExisting" 
              stroke="#8b5cf6" 
              strokeWidth={2}
              name="Mutate Existing"
              dot={{ fill: '#8b5cf6', r: 4 }}
              connectNulls={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Action Comparison */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h4 className="font-semibold mb-3 text-blue-600">Generate New Actions</h4>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-sm">Total Actions:</span>
              <span className="font-medium">{data.generate_new.length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm">Average Reward:</span>
              <span className="font-medium">{generateNewAvg.toFixed(4)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm">Best Reward:</span>
              <span className="font-medium">
                {data.generate_new.length > 0 ? Math.max(...data.generate_new).toFixed(4) : 'N/A'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm">Worst Reward:</span>
              <span className="font-medium">
                {data.generate_new.length > 0 ? Math.min(...data.generate_new).toFixed(4) : 'N/A'}
              </span>
            </div>
          </div>
        </div>

        <div>
          <h4 className="font-semibold mb-3 text-purple-600">Mutate Existing Actions</h4>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-sm">Total Actions:</span>
              <span className="font-medium">{data.mutate_existing.length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm">Average Reward:</span>
              <span className="font-medium">{mutateExistingAvg.toFixed(4)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm">Best Reward:</span>
              <span className="font-medium">
                {data.mutate_existing.length > 0 ? Math.max(...data.mutate_existing).toFixed(4) : 'N/A'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm">Worst Reward:</span>
              <span className="font-medium">
                {data.mutate_existing.length > 0 ? Math.min(...data.mutate_existing).toFixed(4) : 'N/A'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* RL Learning Insights */}
      <div className="p-4 bg-slate-50 dark:bg-slate-800 rounded-lg">
        <h4 className="font-semibold mb-2">🧠 Learning Insights</h4>
        <div className="text-sm space-y-1">
          {generateNewAvg > mutateExistingAvg ? (
            <p>✅ The agent is learning that <strong>generating new features</strong> tends to be more rewarding</p>
          ) : mutateExistingAvg > generateNewAvg ? (
            <p>✅ The agent is learning that <strong>mutating existing features</strong> tends to be more rewarding</p>
          ) : (
            <p>⚖️ Both actions are performing equally well on average</p>
          )}
          
          {generateNewTrend > 0 && (
            <p>📈 Generate new rewards are trending <strong>upward</strong></p>
          )}
          {mutateExistingTrend > 0 && (
            <p>📈 Mutate existing rewards are trending <strong>upward</strong></p>
          )}
          
          {data.generate_new.length === 0 && (
            <p>⏳ Generate new action hasn't been tried yet</p>
          )}
          {data.mutate_existing.length === 0 && (
            <p>⏳ Mutate existing action hasn't been tried yet (need population first)</p>
          )}
        </div>
      </div>
    </div>
  )
} 