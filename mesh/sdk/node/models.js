'use strict'

/** Full advertised model inventory, queried through the embedded native node. */
class Models {
  constructor(handle) {
    this._handle = handle
  }

  async list(query = {}) {
    const { workloadClass, ...filters } = query
    return JSON.parse(await this._handle.meshModelsJson(JSON.stringify({
      ...filters,
      workload_class: workloadClass
    })))
  }

  async get(id) {
    return JSON.parse(await this._handle.meshModelJson(id))
  }
}

module.exports = { Models }
