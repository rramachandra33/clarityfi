from __future__ import annotations

from typing import Any

import httpx


class OpenAIError(RuntimeError):
    pass


class OpenAIClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        chat_model: str,
        embed_model: str,
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.timeout = timeout

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text[:300].strip()

        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"][:300].strip()
        return response.text[:300].strip()

    def _status_error(
        self,
        exc: httpx.HTTPStatusError,
        operation: str,
        model: str,
    ) -> OpenAIError:
        status = exc.response.status_code
        detail = self._error_detail(exc.response)

        if status == 401:
            message = "OpenAI authentication failed. Check OPENAI_API_KEY."
        elif status == 403:
            message = "The OpenAI API key does not have permission for this request."
        elif status == 404:
            message = f"OpenAI model '{model}' was not found or is unavailable."
        elif status == 429:
            message = "OpenAI rate limit, usage limit, or quota exceeded."
        elif status >= 500:
            message = "OpenAI is temporarily unavailable."
        else:
            message = f"{operation} failed with model '{model}'."

        if detail:
            message = f"{message} {detail}"
        return OpenAIError(message)

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            with httpx.Client(trust_env=False, timeout=5.0) as client:
                response = client.get(
                    f"{self.base_url}/models",
                    headers=self._headers,
                )
                response.raise_for_status()
                return True
        except (httpx.HTTPError, OSError):
            return False

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise OpenAIError("Every embedding input must be a non-empty string.")

        try:
            with httpx.Client(trust_env=False, timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/embeddings",
                    headers=self._headers,
                    json={
                        "model": self.embed_model,
                        "input": texts,
                        "encoding_format": "float",
                    },
                )
                response.raise_for_status()
                payload: Any = response.json()

            data = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(data, list) or len(data) != len(texts):
                raise OpenAIError("OpenAI returned an unexpected embeddings response.")

            try:
                ordered = sorted(data, key=lambda item: item["index"])
            except (KeyError, TypeError) as exc:
                raise OpenAIError("OpenAI returned invalid embedding indexes.") from exc

            embeddings: list[list[float]] = []
            for item in ordered:
                vector = item.get("embedding") if isinstance(item, dict) else None
                if not isinstance(vector, list) or not vector:
                    raise OpenAIError("OpenAI returned an invalid embedding vector.")
                embeddings.append(vector)
            return embeddings

        except httpx.TimeoutException as exc:
            raise OpenAIError("The OpenAI embedding request timed out.") from exc
        except httpx.ConnectError as exc:
            raise OpenAIError("Could not connect to the OpenAI API.") from exc
        except httpx.HTTPStatusError as exc:
            raise self._status_error(exc, "Embedding", self.embed_model) from exc
        except httpx.RequestError as exc:
            raise OpenAIError(f"OpenAI embedding request failed: {exc}") from exc
        except ValueError as exc:
            raise OpenAIError("OpenAI returned invalid JSON for the embedding request.") from exc
