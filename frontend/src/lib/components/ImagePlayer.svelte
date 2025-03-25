<script lang="ts">
  import { lcmLiveStatus, LCMLiveStatus, streamId } from '$lib/lcmLive';
  import { getPipelineValues } from '$lib/store';

  import Button from '$lib/components/Button.svelte';
  import Floppy from '$lib/icons/floppy.svelte';
  import Expand from '$lib/icons/expand.svelte';
  import { snapImage, expandWindow } from '$lib/utils';

  $: isLCMRunning =
    $lcmLiveStatus !== LCMLiveStatus.DISCONNECTED && $lcmLiveStatus !== LCMLiveStatus.ERROR;
  let imageEl: HTMLImageElement;
  let expandedWindow: Window;
  let isExpanded = false;
  async function takeSnapshot() {
    if (isLCMRunning) {
      await snapImage(imageEl, {
        prompt: getPipelineValues()?.prompt,
        negative_prompt: getPipelineValues()?.negative_prompt,
        seed: getPipelineValues()?.seed,
        guidance_scale: getPipelineValues()?.guidance_scale
      });
    }
  }
  async function toggleFullscreen() {
    if (isLCMRunning && !isExpanded) {
      expandedWindow = expandWindow('/api/stream/' + $streamId);
      expandedWindow.addEventListener('beforeunload', () => {
        isExpanded = false;
      });
      isExpanded = true;
    } else {
      expandedWindow?.close();
      isExpanded = false;
    }
  }
</script>

<div
  class="relative mx-auto aspect-square max-w-lg self-center overflow-hidden rounded-lg border border-slate-300"
>
  <!-- svelte-ignore a11y-missing-attribute -->
  {#if $lcmLiveStatus === LCMLiveStatus.CONNECTING}
    <!-- Show connecting spinner -->
    <div class="flex h-full w-full items-center justify-center">
      <div class="h-16 w-16 animate-spin rounded-full border-b-2 border-white"></div>
      <p class="ml-2 text-white">Connecting...</p>
    </div>
  {:else if isLCMRunning}
    {#if !isExpanded}
      <!-- Handle image error by adding onerror event -->
      <img
        bind:this={imageEl}
        class="aspect-square w-full rounded-lg"
        src={'/api/stream/' + $streamId}
        on:error={(e) => {
          console.error('Image stream error:', e);
          // If stream fails to load, set status to error
          if ($lcmLiveStatus !== LCMLiveStatus.ERROR) {
            lcmLiveStatus.set(LCMLiveStatus.ERROR);
          }
        }}
      />
    {/if}
    <div class="absolute bottom-1 right-1">
      <Button
        on:click={toggleFullscreen}
        title={'Expand Fullscreen'}
        classList={'text-sm ml-auto text-white p-1 shadow-lg rounded-lg opacity-50'}
      >
        <Expand classList={''} />
      </Button>
      <Button
        on:click={takeSnapshot}
        disabled={!isLCMRunning}
        title={'Take Snapshot'}
        classList={'text-sm ml-auto text-white p-1 shadow-lg rounded-lg opacity-50'}
      >
        <Floppy classList={''} />
      </Button>
    </div>
  {:else if $lcmLiveStatus === LCMLiveStatus.ERROR}
    <!-- Show error state with red border -->
    <div
      class="flex h-full w-full items-center justify-center rounded-lg border-2 border-red-500 bg-gray-900"
    >
      <p class="p-4 text-center text-white">Connection error</p>
    </div>
  {:else}
    <img
      class="aspect-square w-full rounded-lg"
      src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    />
  {/if}
</div>
