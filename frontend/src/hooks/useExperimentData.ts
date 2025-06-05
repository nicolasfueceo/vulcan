'use client'

import { useState, useCallback, useEffect } from 'react'
import { VisualizationData, EvolutionData, Experiment, LLMInteractionLog } from '@/types/vulcan'

interface ExperimentListItem {
  id: string
  name: string
  start_time: string | null
  status: string
}

interface UseExperimentDataResult {
  experiments: ExperimentListItem[]
  selectedExperiment: string | null
  setSelectedExperiment: (id: string | null) => void
  experimentData: EvolutionData | null
  llmLogs: LLMInteractionLog[]
  isLoading: boolean
  error: string | null
  refreshData: () => void
}

const API_BASE_URL = 'http://localhost:8000/api'

export function useExperimentData(): UseExperimentDataResult {
  const [experiments, setExperiments] = useState<ExperimentListItem[]>([])
  const [selectedExperiment, setSelectedExperiment] = useState<string | null>(null)
  const [experimentData, setExperimentData] = useState<EvolutionData | null>(null)
  const [llmLogs, setLlmLogs] = useState<LLMInteractionLog[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchExperiments = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/experiments/list`)
      if (!response.ok) {
        throw new Error('Failed to fetch experiment list.')
      }
      
      const result = await response.json()
      const loadedExperiments = result.experiments || []
      setExperiments(loadedExperiments)

      if (loadedExperiments.length > 0 && !selectedExperiment) {
        setSelectedExperiment(loadedExperiments[0].id)
      }
    } catch (e: any) {
      setError(e.message)
    }
  }, [selectedExperiment])

  const fetchExperimentDetail = useCallback(async (experimentName: string) => {
    setIsLoading(true)
    setError(null)
    setLlmLogs([])

    try {
      const [dataRes, logsRes] = await Promise.all([
        fetch(`${API_BASE_URL}/experiments/${experimentName}/data`),
        fetch(`${API_BASE_URL}/experiments/${experimentName}/llm-logs`),
      ])

      if (!dataRes.ok) throw new Error(`Failed to fetch data for experiment: ${experimentName}`)
      if (!logsRes.ok) throw new Error(`Failed to fetch LLM logs for experiment: ${experimentName}`)
      
      const dataResult = await dataRes.json()
      const logsResult = await logsRes.json()

      if (dataResult.status === 'success') setExperimentData(dataResult.data)
      else throw new Error(dataResult.message || 'Failed to parse experiment data.')

      if (logsResult.status === 'success') setLlmLogs(logsResult.data)
      else throw new Error(logsResult.message || 'Failed to parse LLM logs.')

    } catch (e: any) {
      setError(e.message)
    } finally {
      setIsLoading(false)
    }
  }, [])

  const refreshData = useCallback(async () => {
    await fetchExperiments()
    if (selectedExperiment) {
      await fetchExperimentDetail(selectedExperiment)
    } else if (experiments.length > 0) {
      // If no experiment is selected, but we have a list, select the first one.
      await fetchExperimentDetail(experiments[0].id)
    } else {
      // No experiments available at all
      setExperimentData(null)
      setLlmLogs([])
      setIsLoading(false)
    }
  }, [fetchExperiments, fetchExperimentDetail, selectedExperiment, experiments])

  useEffect(() => {
    fetchExperiments()
  }, [])

  useEffect(() => {
    if (selectedExperiment) {
      fetchExperimentDetail(selectedExperiment)
    }
  }, [selectedExperiment, fetchExperimentDetail])

  return {
    experiments,
    selectedExperiment,
    setSelectedExperiment,
    experimentData,
    llmLogs,
    isLoading,
    error,
    refreshData,
  }
} 