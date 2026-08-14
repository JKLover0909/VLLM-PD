export function quickAnswerRequestOptions(suggestion) {
  if (
    suggestion?.execution === "server_prepared" &&
    typeof suggestion.id === "string" &&
    suggestion.id.trim()
  ) {
    return { quickAnswerId: suggestion.id };
  }
  return { quickAnswerId: null };
}
