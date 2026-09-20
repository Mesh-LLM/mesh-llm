use super::*;
use crate::provisioning::WalletFactory;

struct Factory {
    opens: AtomicUsize,
    wallet: Arc<MockWallet>,
}

#[async_trait]
impl WalletFactory for Factory {
    fn is_provisioned(&self, directory: &std::path::Path) -> bool {
        directory.join("fake-provider-ready").exists()
    }

    async fn open(&self, directory: &std::path::Path) -> Result<Arc<dyn WalletProvider>> {
        self.opens.fetch_add(1, Ordering::SeqCst);
        std::fs::write(directory.join("fake-provider-ready"), b"ready")?;
        Ok(self.wallet.clone())
    }
}

#[tokio::test]
async fn factory_is_lazy_single_flight_and_survives_service_restart() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let factory = Arc::new(Factory {
        opens: AtomicUsize::new(0),
        wallet: Arc::default(),
    });
    {
        let service = PaymentService::with_factory(directory.path(), factory.clone())?;
        assert!(!service.has_wallet());
        assert_eq!(factory.opens.load(Ordering::SeqCst), 0);
        let (first, second) = tokio::join!(service.wallet(), service.wallet());
        assert!(Arc::ptr_eq(first?, second?));
        assert!(service.has_wallet());
        assert_eq!(factory.opens.load(Ordering::SeqCst), 1);
    }
    let service = PaymentService::with_factory(directory.path(), factory.clone())?;
    assert!(service.has_wallet());
    assert_eq!(factory.opens.load(Ordering::SeqCst), 1);
    service.wallet().await?;
    assert_eq!(factory.opens.load(Ordering::SeqCst), 2);
    assert!(!directory.path().join("lexe").exists());
    Ok(())
}
