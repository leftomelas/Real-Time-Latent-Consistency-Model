<script lang="ts">
  export let message: string = '';

  let timeout = 0;
  $: if (message !== '') {
    console.log('message', message);
    clearTimeout(timeout);
    timeout = setTimeout(() => {
      message = '';
    }, 5000);
  }
</script>

{#if message}
  <div role="alert" class="fixed right-0 top-0 m-4">
    <button
      type="button"
      class="w-full"
      on:click={() => (message = '')}
      on:keydown={(e) => e.key === 'Enter' && (message = '')}
    >
      <div class="rounded bg-red-800 p-4 text-white">
        {message}
      </div>
      <div class="bar transition-all duration-500" style="width: 0;"></div>
    </button>
  </div>
{/if}

<style lang="postcss" scoped>
</style>
