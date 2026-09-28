// Passes the form's answers to the model and its answer back. Keeps and logs nothing.
// On Vercel it calls the SageMaker endpoint with a role it gets through Vercel's OIDC (no stored
// AWS keys). Without SAGEMAKER_ENDPOINT it calls the local container, for development.
import { InvokeEndpointCommand, SageMakerRuntimeClient } from "@aws-sdk/client-sagemaker-runtime";
import { awsCredentialsProvider } from "@vercel/oidc-aws-credentials-provider";

const ENDPOINT = process.env.SAGEMAKER_ENDPOINT;
const MODEL_URL = process.env.MODEL_URL || "http://localhost:8080/invocations";
const client = ENDPOINT && new SageMakerRuntimeClient({
  region: process.env.AWS_REGION,
  credentials: awsCredentialsProvider({ roleArn: process.env.AWS_ROLE_ARN }),
});

export async function POST(request) {
  const body = await request.text();
  if (body.length > 2000) return Response.json({ error: "too large" }, { status: 413 });
  try {
    if (!ENDPOINT) {
      const response = await fetch(MODEL_URL, {
        method: "POST", headers: { "Content-Type": "application/json" }, body,
      });
      return Response.json(await response.json(), { status: response.status });
    }
    const result = await client.send(new InvokeEndpointCommand({
      EndpointName: ENDPOINT, ContentType: "application/json", Body: body,
    }));
    return Response.json(JSON.parse(new TextDecoder().decode(result.Body)));
  } catch (error) {
    // The container's own "bad answers" reply comes back from SageMaker as a ModelError.
    if (error.name === "ModelError" && error.OriginalStatusCode === 400) {
      try {
        return Response.json(JSON.parse(error.OriginalMessage), { status: 400 });
      } catch {}
    }
    console.error("model call failed:", error.name);
    return Response.json({ error: "the model did not answer" }, { status: 502 });
  }
}
