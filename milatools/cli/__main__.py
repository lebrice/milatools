from .commands import main
import sentry_sdk


if __name__ == "__main__":
    sentry_sdk.init(
        dsn="https://9316837d50a10b9354c8614c931ed6b8@o4506621497835520.ingest.sentry.io/4506621501636608",
        # Set traces_sample_rate to 1.0 to capture 100%
        # of transactions for performance monitoring.
        traces_sample_rate=1.0,
        # Set profiles_sample_rate to 1.0 to profile 100%
        # of sampled transactions.
        # We recommend adjusting this value in production.
        profiles_sample_rate=1.0,
    )
    main()
