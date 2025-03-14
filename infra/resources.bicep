@description('The location used for all deployed resources')
param location string = resourceGroup().location

@description('Tags that will be applied to all resources')
param tags object = {}


param xnl21bai1224Llm4Exists bool
@secure()
param xnl21bai1224Llm4Definition object

@description('Id of the user or app to assign application roles')
param principalId string

var abbrs = loadJsonContent('./abbreviations.json')
var resourceToken = uniqueString(subscription().id, resourceGroup().id, location)

// Monitor application with Azure Monitor
module monitoring 'br/public:avm/ptn/azd/monitoring:0.1.0' = {
  name: 'monitoring'
  params: {
    logAnalyticsName: '${abbrs.operationalInsightsWorkspaces}${resourceToken}'
    applicationInsightsName: '${abbrs.insightsComponents}${resourceToken}'
    applicationInsightsDashboardName: '${abbrs.portalDashboards}${resourceToken}'
    location: location
    tags: tags
  }
}

// Container registry
module containerRegistry 'br/public:avm/res/container-registry/registry:0.1.1' = {
  name: 'registry'
  params: {
    name: '${abbrs.containerRegistryRegistries}${resourceToken}'
    location: location
    tags: tags
    publicNetworkAccess: 'Enabled'
    roleAssignments:[
      {
        principalId: xnl21bai1224Llm4Identity.outputs.principalId
        principalType: 'ServicePrincipal'
        roleDefinitionIdOrName: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
      }
    ]
  }
}

// Container apps environment
module containerAppsEnvironment 'br/public:avm/res/app/managed-environment:0.4.5' = {
  name: 'container-apps-environment'
  params: {
    logAnalyticsWorkspaceResourceId: monitoring.outputs.logAnalyticsWorkspaceResourceId
    name: '${abbrs.appManagedEnvironments}${resourceToken}'
    location: location
    zoneRedundant: false
  }
}

module xnl21bai1224Llm4Identity 'br/public:avm/res/managed-identity/user-assigned-identity:0.2.1' = {
  name: 'xnl21bai1224Llm4identity'
  params: {
    name: '${abbrs.managedIdentityUserAssignedIdentities}xnl21bai1224Llm4-${resourceToken}'
    location: location
  }
}

module xnl21bai1224Llm4FetchLatestImage './modules/fetch-container-image.bicep' = {
  name: 'xnl21bai1224Llm4-fetch-image'
  params: {
    exists: xnl21bai1224Llm4Exists
    name: 'xnl-21bai1224-llm-4'
  }
}

var xnl21bai1224Llm4AppSettingsArray = filter(array(xnl21bai1224Llm4Definition.settings), i => i.name != '')
var xnl21bai1224Llm4Secrets = map(filter(xnl21bai1224Llm4AppSettingsArray, i => i.?secret != null), i => {
  name: i.name
  value: i.value
  secretRef: i.?secretRef ?? take(replace(replace(toLower(i.name), '_', '-'), '.', '-'), 32)
})
var xnl21bai1224Llm4Env = map(filter(xnl21bai1224Llm4AppSettingsArray, i => i.?secret == null), i => {
  name: i.name
  value: i.value
})

module xnl21bai1224Llm4 'br/public:avm/res/app/container-app:0.8.0' = {
  name: 'xnl21bai1224Llm4'
  params: {
    name: 'xnl-21bai1224-llm-4'
    ingressTargetPort: 80
    scaleMinReplicas: 1
    scaleMaxReplicas: 10
    secrets: {
      secureList:  union([
      ],
      map(xnl21bai1224Llm4Secrets, secret => {
        name: secret.secretRef
        value: secret.value
      }))
    }
    containers: [
      {
        image: xnl21bai1224Llm4FetchLatestImage.outputs.?containers[?0].?image ?? 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'
        name: 'main'
        resources: {
          cpu: json('0.5')
          memory: '1.0Gi'
        }
        env: union([
          {
            name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
            value: monitoring.outputs.applicationInsightsConnectionString
          }
          {
            name: 'AZURE_CLIENT_ID'
            value: xnl21bai1224Llm4Identity.outputs.clientId
          }
          {
            name: 'PORT'
            value: '80'
          }
        ],
        xnl21bai1224Llm4Env,
        map(xnl21bai1224Llm4Secrets, secret => {
            name: secret.name
            secretRef: secret.secretRef
        }))
      }
    ]
    managedIdentities:{
      systemAssigned: false
      userAssignedResourceIds: [xnl21bai1224Llm4Identity.outputs.resourceId]
    }
    registries:[
      {
        server: containerRegistry.outputs.loginServer
        identity: xnl21bai1224Llm4Identity.outputs.resourceId
      }
    ]
    environmentResourceId: containerAppsEnvironment.outputs.resourceId
    location: location
    tags: union(tags, { 'azd-service-name': 'xnl-21bai1224-llm-4' })
  }
}
output AZURE_CONTAINER_REGISTRY_ENDPOINT string = containerRegistry.outputs.loginServer
output AZURE_RESOURCE_XNL_21BAI1224_LLM_4_ID string = xnl21bai1224Llm4.outputs.resourceId
