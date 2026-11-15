// Shapes match services/inference_api/main.py
export interface Health {
  status: string
  model_loaded: boolean
  device: string
  disclaimer: string
}

export interface Study {
  id: string
  patient_id: string | null
  study_date: string | null
  description: string | null
}

export interface StudiesResponse {
  studies: Study[]
  disclaimer: string
}

export interface Prediction {
  instance_id: string
  predicted_class: string
  pneumonia_probability: number
  threshold: number
  disclaimer: string
  gradcam_png_base64?: string
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  if (!res.ok) {
    let detail = ''
    try {
      detail = (await res.json()).detail ?? ''
    } catch {
      // response had no JSON body
    }
    throw new Error(detail || `Request failed (${res.status})`)
  }
  return res.json() as Promise<T>
}

const get = <T>(path: string) => request<T>(`/api${path}`)

export const getHealth = () => get<Health>('/health')
export const getStudies = () => get<StudiesResponse>('/orthanc/studies')

// Orthanc is reached through the dev proxy; the study ID is not an instance ID
export async function getFirstInstanceId(studyId: string): Promise<string> {
  const instances = await request<{ ID: string }[]>(`/orthanc-proxy/studies/${encodeURIComponent(studyId)}/instances`)
  if (instances.length === 0) throw new Error('This study has no instances')
  return instances[0].ID
}

export const getPreviewUrl = (instanceId: string) => `/orthanc-proxy/instances/${encodeURIComponent(instanceId)}/preview`

export const predict = (instanceId: string) =>
  request<Prediction>(`/api/predict/${encodeURIComponent(instanceId)}?gradcam=true`, { method: 'POST' })
